"""Community detection, summarization, and embedding processors."""

from typing import Any, Dict, List, Optional
import logging
import uuid

from ..core.base import BaseProcessor
from ..core.registry import processor_registry
from ..store import create_store
from ..llm.openai_client import OpenAIClient
from ..modeling.embeddings import create_embedding_model
from ..store.vector import create_vector_store

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
except ImportError:
    nx = None
    louvain_communities = None

logger = logging.getLogger(__name__)


class CommunityDetectorProcessor(BaseProcessor):
    """Detect communities in the knowledge graph at one or more hierarchical levels.

    Level 0 uses the graph store's native ``detect_communities()`` (GDS/MAGE).
    Higher levels build a weighted inter-community meta-graph and apply
    ``networkx.community.louvain_communities()`` for portable detection.

    Config keys:
        method: str — "louvain" or "leiden" (default: "louvain").
        levels: int — number of hierarchical levels (default: 1).
        resolution: float — resolution parameter for Louvain (default: 1.0).
        Store params: store_type, neo4j_uri, etc.
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self._store = None
        self.method = config_dict.get("method", "louvain")
        self.levels = config_dict.get("levels", 1)
        self.resolution = config_dict.get("resolution", 1.0)

    def _ensure_store(self):
        if self._store is None:
            self._store = create_store(self.config_dict)

    def __call__(self, **kwargs) -> Dict[str, Any]:
        self._ensure_store()

        all_communities: Dict[int, Dict[str, str]] = {}  # level -> {entity_or_child_id -> community_id}

        # Level 0: use graph store's native community detection
        level0_mapping = self._store.detect_communities(method=self.method)
        all_communities[0] = level0_mapping

        # Write level-0 Community nodes and MEMBER_OF relationships
        community_ids_at_level0 = set(level0_mapping.values())
        for comm_id in community_ids_at_level0:
            self._store.write_community(
                community_id=comm_id, level=0, title="", summary="",
            )
        for entity_id, comm_id in level0_mapping.items():
            self._store.write_community_membership(entity_id, comm_id, level=0)

        logger.info(
            f"Level 0: {len(community_ids_at_level0)} communities, "
            f"{len(level0_mapping)} entities"
        )

        # Higher levels: build meta-graph and run networkx Louvain
        prev_mapping = level0_mapping  # entity_id -> community_id at previous level
        for level in range(1, self.levels):
            prev_community_ids = list(set(prev_mapping.values()))
            if len(prev_community_ids) <= 1:
                logger.info(f"Level {level}: only {len(prev_community_ids)} community, stopping.")
                break

            # Build weighted inter-community edges
            inter_edges = self._store.get_inter_community_edges(prev_mapping)
            if not inter_edges:
                logger.info(f"Level {level}: no inter-community edges, stopping.")
                break

            # Use networkx for higher-level detection
            if nx is None:
                raise ImportError(
                    "networkx package required for multi-level community detection. "
                    "Install with: pip install networkx"
                )

            G = nx.Graph()
            G.add_nodes_from(prev_community_ids)
            for a, b, w in inter_edges:
                G.add_edge(a, b, weight=w)

            nx_communities = louvain_communities(
                G, weight="weight", resolution=self.resolution, seed=42,
            )

            # Map child community -> parent community
            level_mapping: Dict[str, str] = {}
            for community_set in nx_communities:
                parent_id = str(uuid.uuid4())
                self._store.write_community(
                    community_id=parent_id, level=level, title="", summary="",
                )
                for child_id in community_set:
                    level_mapping[child_id] = parent_id
                    self._store.write_community_hierarchy(child_id, parent_id)

            all_communities[level] = level_mapping

            logger.info(
                f"Level {level}: {len(set(level_mapping.values()))} communities "
                f"from {len(prev_community_ids)} children"
            )

            # For next level, remap entity_id -> new parent community
            new_prev: Dict[str, str] = {}
            for entity_id, l0_comm in prev_mapping.items():
                parent = level_mapping.get(l0_comm, l0_comm)
                new_prev[entity_id] = parent
            prev_mapping = new_prev

        total_communities = sum(
            len(set(m.values())) for m in all_communities.values()
        )
        return {
            "communities": all_communities,
            "community_count": total_communities,
            "levels": len(all_communities),
        }


class CommunitySummarizerProcessor(BaseProcessor):
    """Generate LLM summaries for communities using top-k entities by degree.

    Config keys:
        top_k: int — max entities per community for context (default: 10).
        api_key: str — OpenAI API key.
        model: str — LLM model name (default: "gpt-4o-mini").
        base_url: str — optional custom API base URL.
        temperature: float — LLM temperature (default: 0.3).
        max_completion_tokens: int — max tokens (default: 4096).
        Store params: store_type, neo4j_uri, etc.
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self._store = None
        self._llm = None
        self.top_k = config_dict.get("top_k", 10)

    def _ensure_store(self):
        if self._store is None:
            self._store = create_store(self.config_dict)

    def _ensure_llm(self):
        if self._llm is None:
            self._llm = OpenAIClient(
                api_key=self.config_dict.get("api_key"),
                base_url=self.config_dict.get("base_url"),
                model=self.config_dict.get("model", "gpt-4o-mini"),
                temperature=self.config_dict.get("temperature", 0.3),
                max_completion_tokens=self.config_dict.get("max_completion_tokens", 4096),
            )

    def __call__(self, **kwargs) -> Dict[str, Any]:
        self._ensure_store()
        self._ensure_llm()

        communities = self._store.get_all_communities()
        summaries: Dict[str, Dict[str, str]] = {}

        for comm in communities:
            comm_id = comm.get("id")
            if not comm_id:
                continue

            # Get member entities
            members = self._store.get_community_members(comm_id)
            if not members:
                continue

            # Get top-k by degree centrality
            member_ids = [m.get("id") for m in members if m.get("id")]
            top_entities = self._store.get_top_entities_by_degree(
                entity_ids=member_ids, top_k=self.top_k,
            )

            # Build context for LLM
            context = self._build_context(top_entities)
            if not context.strip():
                continue

            title, summary = self._generate_summary(context)

            # Update community in store
            level = comm.get("level", 0)
            self._store.write_community(comm_id, level, title, summary)
            summaries[comm_id] = {"title": title, "summary": summary}

        logger.info(f"Summarized {len(summaries)} communities")
        return {
            "summaries": summaries,
            "summarized_count": len(summaries),
        }

    def _build_context(self, entities: List[Dict[str, Any]]) -> str:
        """Build textual context from top entities and their relations."""
        lines = []
        for ent in entities:
            label = ent.get("label", "unknown")
            etype = ent.get("entity_type", "")
            degree = ent.get("degree", 0)
            line = f"- {label} (type: {etype}, connections: {degree})"
            lines.append(line)

        # Fetch relations between these entities for richer context
        entity_ids = [e.get("id") for e in entities if e.get("id")]
        for eid in entity_ids[:5]:  # limit relation queries
            try:
                rels = self._store.get_entity_relations(eid)
                for rel in rels[:3]:  # limit per entity
                    lines.append(
                        f"  {rel.get('target_label', '?')} --[{rel.get('relation_type', '?')}]--> related"
                    )
            except Exception:
                pass

        return "\n".join(lines)

    def _generate_summary(self, context: str) -> tuple:
        """Use LLM to generate a title and summary for a community."""
        prompt = (
            "You are analyzing a community of entities in a knowledge graph. "
            "Based on the following entities and their relationships, provide:\n"
            "1. A short title (max 10 words) describing the community theme\n"
            "2. A concise summary (2-3 sentences) of what this community represents\n\n"
            f"Entities and relationships:\n{context}\n\n"
            "Respond in this exact format:\n"
            "TITLE: <title>\n"
            "SUMMARY: <summary>"
        )

        response = self._llm.complete([
            {"role": "system", "content": "You are a knowledge graph analyst."},
            {"role": "user", "content": prompt},
        ])

        title = ""
        summary = ""
        for line in response.strip().split("\n"):
            if line.startswith("TITLE:"):
                title = line[len("TITLE:"):].strip()
            elif line.startswith("SUMMARY:"):
                summary = line[len("SUMMARY:"):].strip()

        # Fallback: use full response as summary
        if not title:
            title = response[:50].strip()
        if not summary:
            summary = response.strip()

        return title, summary


class CommunityEmbedderProcessor(BaseProcessor):
    """Embed community summaries and store in vector store + graph.

    Config keys:
        embedding_method: str — "sentence_transformer" or "openai" (default: "sentence_transformer").
        model_name: str — embedding model name (default: "all-MiniLM-L6-v2").
        vector_store_type: str — "in_memory", "faiss", or "qdrant" (default: "in_memory").
        Store params: store_type, neo4j_uri, etc.
        Additional embedding/vector store params passed through.
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self._store = None
        self._embedding_model = None
        self._vector_store = None

    def _ensure_store(self):
        if self._store is None:
            self._store = create_store(self.config_dict)

    def _ensure_embedding_model(self):
        if self._embedding_model is None:
            # Build embedding config from processor config
            embed_config = {
                "embedding_method": self.config_dict.get("embedding_method", "sentence_transformer"),
            }
            # Pass through relevant embedding params
            for key in ("model_name", "device", "batch_size", "api_key", "base_url",
                        "model", "dimensions", "timeout"):
                if key in self.config_dict:
                    embed_config[key] = self.config_dict[key]
            self._embedding_model = create_embedding_model(embed_config)

    def _ensure_vector_store(self):
        if self._vector_store is None:
            vector_config = {
                "vector_store_type": self.config_dict.get("vector_store_type", "in_memory"),
            }
            for key in ("use_gpu", "qdrant_url", "qdrant_api_key", "qdrant_path", "prefer_grpc"):
                if key in self.config_dict:
                    vector_config[key] = self.config_dict[key]
            self._vector_store = create_vector_store(vector_config)

    def __call__(self, **kwargs) -> Dict[str, Any]:
        self._ensure_store()
        self._ensure_embedding_model()
        self._ensure_vector_store()

        communities = self._store.get_all_communities()

        # Filter to communities with summaries
        to_embed = []
        for comm in communities:
            summary = comm.get("summary", "")
            if summary:
                to_embed.append(comm)

        if not to_embed:
            logger.info("No communities with summaries to embed")
            return {"embedded_count": 0, "dimension": 0}

        # Create vector index
        dimension = self._embedding_model.dimension
        index_name = "community_embeddings"
        self._vector_store.create_index(index_name, dimension)

        # Encode summaries
        texts = []
        for comm in to_embed:
            title = comm.get("title", "")
            summary = comm.get("summary", "")
            texts.append(f"{title}. {summary}" if title else summary)

        embeddings = self._embedding_model.encode(texts)

        # Store in vector store
        items = [
            (comm.get("id"), emb)
            for comm, emb in zip(to_embed, embeddings)
        ]
        self._vector_store.store_embeddings(index_name, items)

        # Also store on Community nodes in graph
        for comm, emb in zip(to_embed, embeddings):
            try:
                self._store.update_community_embedding(comm.get("id"), emb)
            except Exception as e:
                logger.debug(f"Could not store embedding on Community node: {e}")

        logger.info(f"Embedded {len(to_embed)} communities (dim={dimension})")
        return {
            "embedded_count": len(to_embed),
            "dimension": dimension,
        }


# -- Processor registration --------------------------------------------------

@processor_registry.register("community_detector")
def create_community_detector(config_dict: dict, pipeline=None):
    return CommunityDetectorProcessor(config_dict, pipeline)


@processor_registry.register("community_summarizer")
def create_community_summarizer(config_dict: dict, pipeline=None):
    return CommunitySummarizerProcessor(config_dict, pipeline)


@processor_registry.register("community_embedder")
def create_community_embedder(config_dict: dict, pipeline=None):
    return CommunityEmbedderProcessor(config_dict, pipeline)
