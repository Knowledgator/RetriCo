"""Tests for community detection, summarization, and embedding pipeline."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import uuid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    """Create a mock graph store with community-related methods."""
    store = MagicMock()
    store.detect_communities.return_value = {
        "e1": "c1",
        "e2": "c1",
        "e3": "c2",
        "e4": "c2",
        "e5": "c3",
    }
    store.get_all_communities.return_value = [
        {"id": "c1", "level": 0, "title": "", "summary": ""},
        {"id": "c2", "level": 0, "title": "", "summary": ""},
        {"id": "c3", "level": 0, "title": "", "summary": ""},
    ]
    store.get_community_members.side_effect = lambda cid: {
        "c1": [{"id": "e1", "label": "Einstein", "entity_type": "person"},
               {"id": "e2", "label": "Bohr", "entity_type": "person"}],
        "c2": [{"id": "e3", "label": "MIT", "entity_type": "org"},
               {"id": "e4", "label": "Stanford", "entity_type": "org"}],
        "c3": [{"id": "e5", "label": "Berlin", "entity_type": "location"}],
    }.get(cid, [])
    store.get_top_entities_by_degree.return_value = [
        {"id": "e1", "label": "Einstein", "entity_type": "person", "degree": 5},
        {"id": "e2", "label": "Bohr", "entity_type": "person", "degree": 3},
    ]
    store.get_entity_relations.return_value = [
        {"relation_type": "WORKS_AT", "target_label": "MIT", "score": 0.9},
    ]
    store.get_inter_community_edges.return_value = [
        ("c1", "c2", 3),
        ("c2", "c3", 1),
    ]
    return store


@pytest.fixture
def mock_llm():
    """Mock LLM client."""
    llm = MagicMock()
    llm.complete.return_value = "TITLE: Physics Researchers\nSUMMARY: A community of notable physicists and their collaborations."
    return llm


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model."""
    model = MagicMock()
    type(model).dimension = PropertyMock(return_value=384)
    model.encode.return_value = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
    return model


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    return MagicMock()


# ---------------------------------------------------------------------------
# CommunityDetectorProcessor
# ---------------------------------------------------------------------------

class TestCommunityDetector:
    """Tests for CommunityDetectorProcessor."""

    def test_single_level_detection(self, mock_store):
        """Test basic single-level community detection."""
        from grapsit.modeling.community import CommunityDetectorProcessor

        config = {"store_type": "neo4j", "method": "louvain", "levels": 1}
        processor = CommunityDetectorProcessor(config)
        processor._store = mock_store

        result = processor()

        assert result["levels"] == 1
        assert result["community_count"] == 3
        assert 0 in result["communities"]
        assert result["communities"][0] == {"e1": "c1", "e2": "c1", "e3": "c2", "e4": "c2", "e5": "c3"}

        # Verify store calls
        mock_store.detect_communities.assert_called_once_with(method="louvain")
        assert mock_store.write_community.call_count == 3
        assert mock_store.write_community_membership.call_count == 5

    def test_leiden_method(self, mock_store):
        """Test detection with leiden method."""
        from grapsit.modeling.community import CommunityDetectorProcessor

        config = {"store_type": "neo4j", "method": "leiden", "levels": 1}
        processor = CommunityDetectorProcessor(config)
        processor._store = mock_store

        processor()

        mock_store.detect_communities.assert_called_once_with(method="leiden")

    @patch("grapsit.modeling.community.uuid")
    def test_multi_level_detection(self, mock_uuid, mock_store):
        """Test hierarchical multi-level detection."""
        from grapsit.modeling.community import CommunityDetectorProcessor

        # Mock uuid to produce predictable IDs
        parent_ids = [f"parent_{i}" for i in range(10)]
        mock_uuid.uuid4.side_effect = [type("", (), {"__str__": lambda s: pid})() for pid in parent_ids]

        config = {"store_type": "neo4j", "method": "louvain", "levels": 2, "resolution": 1.0}
        processor = CommunityDetectorProcessor(config)
        processor._store = mock_store

        # Mock networkx
        with patch("grapsit.modeling.community.nx") as mock_nx, \
             patch("grapsit.modeling.community.louvain_communities") as mock_louvain:
            mock_G = MagicMock()
            mock_nx.Graph.return_value = mock_G
            # Return 2 higher-level communities from 3 level-0 communities
            mock_louvain.return_value = [{"c1", "c2"}, {"c3"}]

            result = processor()

        assert result["levels"] == 2
        assert 0 in result["communities"]
        assert 1 in result["communities"]

        # Level 1 should have written parent communities + hierarchy
        assert mock_store.write_community_hierarchy.call_count >= 2

    def test_multi_level_stops_if_single_community(self, mock_store):
        """Multi-level stops early if only one community at a level."""
        from grapsit.modeling.community import CommunityDetectorProcessor

        # Override to return single community
        mock_store.detect_communities.return_value = {"e1": "c1", "e2": "c1"}

        config = {"store_type": "neo4j", "method": "louvain", "levels": 3}
        processor = CommunityDetectorProcessor(config)
        processor._store = mock_store

        result = processor()

        assert result["levels"] == 1  # Stopped at level 0

    def test_multi_level_stops_if_no_inter_edges(self, mock_store):
        """Multi-level stops if no inter-community edges."""
        from grapsit.modeling.community import CommunityDetectorProcessor

        mock_store.get_inter_community_edges.return_value = []

        config = {"store_type": "neo4j", "method": "louvain", "levels": 2}
        processor = CommunityDetectorProcessor(config)
        processor._store = mock_store

        result = processor()

        assert result["levels"] == 1

    def test_lazy_store_creation(self):
        """Store is created lazily on first call."""
        from grapsit.modeling.community import CommunityDetectorProcessor

        config = {"store_type": "neo4j", "method": "louvain", "levels": 1}
        processor = CommunityDetectorProcessor(config)

        assert processor._store is None

        with patch("grapsit.modeling.community.create_store") as mock_create:
            mock_store = MagicMock()
            mock_store.detect_communities.return_value = {}
            mock_create.return_value = mock_store
            processor()
            mock_create.assert_called_once_with(config)


# ---------------------------------------------------------------------------
# CommunitySummarizerProcessor
# ---------------------------------------------------------------------------

class TestCommunitySummarizer:
    """Tests for CommunitySummarizerProcessor."""

    def test_summarize_communities(self, mock_store, mock_llm):
        """Test summarization of communities."""
        from grapsit.modeling.community import CommunitySummarizerProcessor

        # Add summaries so we can test the full flow
        mock_store.get_all_communities.return_value = [
            {"id": "c1", "level": 0, "title": "", "summary": ""},
            {"id": "c2", "level": 0, "title": "", "summary": ""},
        ]

        config = {"store_type": "neo4j", "model": "gpt-4o-mini", "top_k": 5}
        processor = CommunitySummarizerProcessor(config)
        processor._store = mock_store
        processor._llm = mock_llm

        result = processor()

        assert result["summarized_count"] == 2
        assert "c1" in result["summaries"]
        assert result["summaries"]["c1"]["title"] == "Physics Researchers"
        assert "physicists" in result["summaries"]["c1"]["summary"]

        # Store should be updated with summaries
        assert mock_store.write_community.call_count == 2

    def test_skip_communities_without_members(self, mock_store, mock_llm):
        """Communities with no members are skipped."""
        from grapsit.modeling.community import CommunitySummarizerProcessor

        mock_store.get_all_communities.return_value = [
            {"id": "c_empty", "level": 0},
        ]
        mock_store.get_community_members.return_value = []

        config = {"store_type": "neo4j"}
        processor = CommunitySummarizerProcessor(config)
        processor._store = mock_store
        processor._llm = mock_llm

        result = processor()

        assert result["summarized_count"] == 0
        mock_llm.complete.assert_not_called()

    def test_top_k_entity_selection(self, mock_store, mock_llm):
        """Top-k parameter limits entities per community."""
        from grapsit.modeling.community import CommunitySummarizerProcessor

        config = {"store_type": "neo4j", "top_k": 3}
        processor = CommunitySummarizerProcessor(config)
        processor._store = mock_store
        processor._llm = mock_llm

        processor()

        # get_top_entities_by_degree should be called with top_k=3
        calls = mock_store.get_top_entities_by_degree.call_args_list
        for call in calls:
            assert call.kwargs.get("top_k") == 3 or call[1].get("top_k") == 3

    def test_llm_response_parsing(self, mock_store):
        """Test various LLM response formats."""
        from grapsit.modeling.community import CommunitySummarizerProcessor

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "TITLE: Test Title\nSUMMARY: Test summary here."

        mock_store.get_all_communities.return_value = [
            {"id": "c1", "level": 0},
        ]

        config = {"store_type": "neo4j"}
        processor = CommunitySummarizerProcessor(config)
        processor._store = mock_store
        processor._llm = mock_llm

        result = processor()

        assert result["summaries"]["c1"]["title"] == "Test Title"
        assert result["summaries"]["c1"]["summary"] == "Test summary here."

    def test_llm_response_fallback(self, mock_store):
        """Fallback when LLM doesn't follow expected format."""
        from grapsit.modeling.community import CommunitySummarizerProcessor

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "This is a free-form response about the community."

        mock_store.get_all_communities.return_value = [
            {"id": "c1", "level": 0},
        ]

        config = {"store_type": "neo4j"}
        processor = CommunitySummarizerProcessor(config)
        processor._store = mock_store
        processor._llm = mock_llm

        result = processor()

        # Should use fallback: truncated response as title, full as summary
        assert result["summaries"]["c1"]["title"] != ""
        assert result["summaries"]["c1"]["summary"] != ""

    def test_lazy_llm_creation(self):
        """LLM client is created lazily."""
        from grapsit.modeling.community import CommunitySummarizerProcessor

        config = {"store_type": "neo4j", "api_key": "test-key", "model": "gpt-4o-mini"}
        processor = CommunitySummarizerProcessor(config)

        assert processor._llm is None

        with patch("grapsit.modeling.community.OpenAIClient") as mock_cls:
            processor._ensure_llm()
            mock_cls.assert_called_once_with(
                api_key="test-key",
                base_url=None,
                model="gpt-4o-mini",
                temperature=0.3,
                max_completion_tokens=4096,
            )


# ---------------------------------------------------------------------------
# CommunityEmbedderProcessor
# ---------------------------------------------------------------------------

class TestCommunityEmbedder:
    """Tests for CommunityEmbedderProcessor."""

    def test_embed_communities(self, mock_store, mock_embedding_model, mock_vector_store):
        """Test embedding of community summaries."""
        from grapsit.modeling.community import CommunityEmbedderProcessor

        # Communities with summaries
        mock_store.get_all_communities.return_value = [
            {"id": "c1", "level": 0, "title": "Physics", "summary": "About physics."},
            {"id": "c2", "level": 0, "title": "CS", "summary": "About CS."},
            {"id": "c3", "level": 0, "title": "", "summary": ""},  # No summary — skipped
        ]

        config = {"store_type": "neo4j", "embedding_method": "sentence_transformer"}
        processor = CommunityEmbedderProcessor(config)
        processor._store = mock_store
        processor._embedding_model = mock_embedding_model
        processor._vector_store = mock_vector_store

        # Only 2 communities have summaries, but encode returns 3 vectors.
        # We need to match.
        mock_embedding_model.encode.return_value = [[0.1] * 384, [0.2] * 384]

        result = processor()

        assert result["embedded_count"] == 2
        assert result["dimension"] == 384

        # Vector store should be used
        mock_vector_store.create_index.assert_called_once_with("community_embeddings", 384)
        mock_vector_store.store_embeddings.assert_called_once()

        # Graph store should get embedding updates
        assert mock_store.update_community_embedding.call_count == 2

    def test_no_summaries_to_embed(self, mock_store, mock_embedding_model, mock_vector_store):
        """Skip embedding if no communities have summaries."""
        from grapsit.modeling.community import CommunityEmbedderProcessor

        mock_store.get_all_communities.return_value = [
            {"id": "c1", "level": 0, "title": "", "summary": ""},
        ]

        config = {"store_type": "neo4j"}
        processor = CommunityEmbedderProcessor(config)
        processor._store = mock_store
        processor._embedding_model = mock_embedding_model
        processor._vector_store = mock_vector_store

        result = processor()

        assert result["embedded_count"] == 0
        mock_embedding_model.encode.assert_not_called()

    def test_embedding_text_includes_title_and_summary(self, mock_store, mock_embedding_model, mock_vector_store):
        """Embedding input should combine title and summary."""
        from grapsit.modeling.community import CommunityEmbedderProcessor

        mock_store.get_all_communities.return_value = [
            {"id": "c1", "level": 0, "title": "Physics", "summary": "About physics."},
        ]
        mock_embedding_model.encode.return_value = [[0.1] * 384]

        config = {"store_type": "neo4j"}
        processor = CommunityEmbedderProcessor(config)
        processor._store = mock_store
        processor._embedding_model = mock_embedding_model
        processor._vector_store = mock_vector_store

        processor()

        texts = mock_embedding_model.encode.call_args[0][0]
        assert texts == ["Physics. About physics."]

    def test_lazy_embedding_model_creation(self):
        """Embedding model is created lazily."""
        from grapsit.modeling.community import CommunityEmbedderProcessor

        config = {"store_type": "neo4j", "embedding_method": "sentence_transformer", "model_name": "test-model"}
        processor = CommunityEmbedderProcessor(config)

        assert processor._embedding_model is None

        with patch("grapsit.modeling.community.create_embedding_model") as mock_create:
            processor._ensure_embedding_model()
            mock_create.assert_called_once()
            call_config = mock_create.call_args[0][0]
            assert call_config["embedding_method"] == "sentence_transformer"
            assert call_config["model_name"] == "test-model"

    def test_lazy_vector_store_creation(self):
        """Vector store is created lazily."""
        from grapsit.modeling.community import CommunityEmbedderProcessor

        config = {"store_type": "neo4j", "vector_store_type": "faiss"}
        processor = CommunityEmbedderProcessor(config)

        assert processor._vector_store is None

        with patch("grapsit.modeling.community.create_vector_store") as mock_create:
            processor._ensure_vector_store()
            mock_create.assert_called_once()
            call_config = mock_create.call_args[0][0]
            assert call_config["vector_store_type"] == "faiss"


# ---------------------------------------------------------------------------
# CommunityConfigBuilder
# ---------------------------------------------------------------------------

class TestCommunityConfigBuilder:
    """Tests for CommunityConfigBuilder."""

    def test_detector_only(self):
        """Config with only detector."""
        from grapsit.core.builders import CommunityConfigBuilder

        builder = CommunityConfigBuilder(name="test")
        builder.detector(method="louvain", levels=2)
        config = builder.get_config()

        assert config["name"] == "test"
        assert len(config["nodes"]) == 1
        assert config["nodes"][0]["processor"] == "community_detector"
        assert config["nodes"][0]["config"]["method"] == "louvain"
        assert config["nodes"][0]["config"]["levels"] == 2

    def test_detector_plus_summarizer(self):
        """Config with detector and summarizer."""
        from grapsit.core.builders import CommunityConfigBuilder

        builder = CommunityConfigBuilder()
        builder.detector(method="louvain", neo4j_uri="bolt://myhost:7687")
        builder.summarizer(api_key="sk-test", top_k=5)
        config = builder.get_config()

        assert len(config["nodes"]) == 2
        summarizer_node = config["nodes"][1]
        assert summarizer_node["processor"] == "community_summarizer"
        assert summarizer_node["requires"] == ["community_detector"]
        # Store params inherited from detector
        assert summarizer_node["config"]["neo4j_uri"] == "bolt://myhost:7687"
        assert summarizer_node["config"]["api_key"] == "sk-test"
        assert summarizer_node["config"]["top_k"] == 5

    def test_full_pipeline(self):
        """Config with all three processors."""
        from grapsit.core.builders import CommunityConfigBuilder

        builder = CommunityConfigBuilder()
        builder.detector(method="louvain")
        builder.summarizer(api_key="sk-test")
        builder.embedder(embedding_method="sentence_transformer", model_name="test-model")
        config = builder.get_config()

        assert len(config["nodes"]) == 3
        embedder_node = config["nodes"][2]
        assert embedder_node["processor"] == "community_embedder"
        assert set(embedder_node["requires"]) == {"community_detector", "community_summarizer"}
        assert embedder_node["config"]["embedding_method"] == "sentence_transformer"
        assert embedder_node["config"]["model_name"] == "test-model"

    def test_embedder_without_summarizer(self):
        """Embedder can run without summarizer."""
        from grapsit.core.builders import CommunityConfigBuilder

        builder = CommunityConfigBuilder()
        builder.detector()
        builder.embedder()
        config = builder.get_config()

        assert len(config["nodes"]) == 2
        embedder_node = config["nodes"][1]
        assert embedder_node["requires"] == ["community_detector"]

    def test_missing_detector_raises(self):
        """Error if detector not configured."""
        from grapsit.core.builders import CommunityConfigBuilder

        builder = CommunityConfigBuilder()
        with pytest.raises(ValueError, match="Detector config required"):
            builder.get_config()

    def test_store_params_inherited(self):
        """Store params flow from detector to summarizer and embedder."""
        from grapsit.core.builders import CommunityConfigBuilder

        builder = CommunityConfigBuilder()
        builder.detector(
            store_type="memgraph",
            memgraph_uri="bolt://mg:7687",
            memgraph_password="secret",
        )
        builder.summarizer(api_key="sk-test")
        builder.embedder()
        config = builder.get_config()

        for node in config["nodes"]:
            assert node["config"]["store_type"] == "memgraph"
            assert node["config"]["memgraph_uri"] == "bolt://mg:7687"

    def test_save_yaml(self, tmp_path):
        """Save config to YAML file."""
        from grapsit.core.builders import CommunityConfigBuilder

        builder = CommunityConfigBuilder(name="test_yaml")
        builder.detector(method="louvain")
        builder.summarizer(api_key="sk-test")

        filepath = tmp_path / "community.yaml"
        builder.save(str(filepath))

        assert filepath.exists()
        import yaml
        with open(filepath) as f:
            loaded = yaml.safe_load(f)
        assert loaded["name"] == "test_yaml"
        assert len(loaded["nodes"]) == 2

    def test_build_creates_executor(self):
        """build() creates a DAGExecutor."""
        from grapsit.core.builders import CommunityConfigBuilder

        builder = CommunityConfigBuilder()
        builder.detector(method="louvain")

        executor = builder.build()
        # Should be a DAGExecutor
        from grapsit.core.dag import DAGExecutor
        assert isinstance(executor, DAGExecutor)


# ---------------------------------------------------------------------------
# Processor registration
# ---------------------------------------------------------------------------

class TestProcessorRegistration:
    """Test that community processors are registered."""

    def test_detector_registered(self):
        from grapsit.core.registry import processor_registry
        assert "community_detector" in processor_registry._factories

    def test_summarizer_registered(self):
        from grapsit.core.registry import processor_registry
        assert "community_summarizer" in processor_registry._factories

    def test_embedder_registered(self):
        from grapsit.core.registry import processor_registry
        assert "community_embedder" in processor_registry._factories


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

class TestDetectCommunitiesConvenience:
    """Tests for the detect_communities() convenience function."""

    def test_detect_communities_basic(self):
        """Basic detect_communities() call builds correct pipeline."""
        import grapsit

        with patch("grapsit.CommunityConfigBuilder") as MockBuilder:
            mock_builder = MagicMock()
            MockBuilder.return_value = mock_builder
            mock_executor = MagicMock()
            mock_builder.build.return_value = mock_executor
            mock_ctx = MagicMock()
            mock_executor.execute.return_value = mock_ctx

            result = grapsit.detect_communities(
                method="louvain",
                levels=2,
                neo4j_uri="bolt://test:7687",
            )

            MockBuilder.assert_called_once_with(name="detect_communities")
            mock_builder.detector.assert_called_once()
            detector_kwargs = mock_builder.detector.call_args
            assert detector_kwargs.kwargs["method"] == "louvain"
            assert detector_kwargs.kwargs["levels"] == 2
            assert detector_kwargs.kwargs["neo4j_uri"] == "bolt://test:7687"

            # No api_key → no summarizer/embedder
            mock_builder.summarizer.assert_not_called()
            mock_builder.embedder.assert_not_called()

    def test_detect_communities_with_summarizer(self):
        """detect_communities() with api_key enables summarizer + embedder."""
        import grapsit

        with patch("grapsit.CommunityConfigBuilder") as MockBuilder:
            mock_builder = MagicMock()
            MockBuilder.return_value = mock_builder
            mock_executor = MagicMock()
            mock_builder.build.return_value = mock_executor
            mock_executor.execute.return_value = MagicMock()

            grapsit.detect_communities(
                api_key="sk-test",
                model="gpt-4o",
                top_k=5,
                embedding_method="openai",
            )

            mock_builder.summarizer.assert_called_once_with(
                api_key="sk-test", model="gpt-4o", top_k=5,
            )
            mock_builder.embedder.assert_called_once_with(
                embedding_method="openai",
                model_name="all-MiniLM-L6-v2",
                vector_store_type="in_memory",
            )


# ---------------------------------------------------------------------------
# Store method tests
# ---------------------------------------------------------------------------

class TestStoreNewMethods:
    """Test new store methods on BaseGraphStore (default NotImplementedError)."""

    def test_base_store_write_community_hierarchy(self):
        from grapsit.store.base import BaseGraphStore
        # Can't instantiate ABC, but can test the method exists
        assert hasattr(BaseGraphStore, "write_community_hierarchy")

    def test_base_store_get_top_entities_by_degree(self):
        from grapsit.store.base import BaseGraphStore
        assert hasattr(BaseGraphStore, "get_top_entities_by_degree")

    def test_base_store_update_community_embedding(self):
        from grapsit.store.base import BaseGraphStore
        assert hasattr(BaseGraphStore, "update_community_embedding")

    def test_base_store_get_inter_community_edges(self):
        from grapsit.store.base import BaseGraphStore
        assert hasattr(BaseGraphStore, "get_inter_community_edges")


class TestNeo4jStoreNewMethods:
    """Test Neo4j store implementation of new methods (mocked driver)."""

    @pytest.fixture
    def neo4j_store(self):
        from grapsit.store.neo4j_store import Neo4jGraphStore
        store = Neo4jGraphStore.__new__(Neo4jGraphStore)
        store.uri = "bolt://test:7687"
        store.user = "neo4j"
        store.password = "password"
        store.database = "neo4j"
        store._driver = MagicMock()
        store._run = MagicMock()
        return store

    def test_write_community_hierarchy(self, neo4j_store):
        neo4j_store.write_community_hierarchy("child1", "parent1")
        neo4j_store._run.assert_called_once()
        call_args = neo4j_store._run.call_args
        assert "CHILD_OF" in call_args[0][0]
        assert call_args[0][1] == {"child_id": "child1", "parent_id": "parent1"}

    def test_get_top_entities_by_degree_all(self, neo4j_store):
        neo4j_store._run.return_value = [
            {"e": {"id": "e1", "label": "A"}, "degree": 5},
        ]
        result = neo4j_store.get_top_entities_by_degree(top_k=10)
        assert len(result) == 1
        assert result[0]["degree"] == 5

    def test_get_top_entities_by_degree_filtered(self, neo4j_store):
        neo4j_store._run.return_value = [
            {"e": {"id": "e1", "label": "A"}, "degree": 5},
        ]
        result = neo4j_store.get_top_entities_by_degree(entity_ids=["e1", "e2"], top_k=5)
        call_args = neo4j_store._run.call_args
        assert "WHERE e.id IN $ids" in call_args[0][0]

    def test_update_community_embedding(self, neo4j_store):
        neo4j_store.update_community_embedding("c1", [0.1, 0.2, 0.3])
        call_args = neo4j_store._run.call_args
        assert "SET co.embedding = $embedding" in call_args[0][0]

    def test_get_inter_community_edges(self, neo4j_store):
        neo4j_store._run.return_value = [
            {"head_id": "e1", "tail_id": "e3"},
            {"head_id": "e2", "tail_id": "e3"},
        ]
        memberships = {"e1": "c1", "e2": "c1", "e3": "c2"}
        result = neo4j_store.get_inter_community_edges(memberships)
        assert len(result) == 1
        assert result[0][2] == 2  # weight = 2 edges between c1 and c2

    def test_inter_community_edges_skip_same_community(self, neo4j_store):
        neo4j_store._run.return_value = [
            {"head_id": "e1", "tail_id": "e2"},  # same community
        ]
        memberships = {"e1": "c1", "e2": "c1"}
        result = neo4j_store.get_inter_community_edges(memberships)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------

class TestPublicAPI:
    """Test that new items are properly exported."""

    def test_community_config_builder_exported(self):
        import grapsit
        assert hasattr(grapsit, "CommunityConfigBuilder")

    def test_detect_communities_exported(self):
        import grapsit
        assert hasattr(grapsit, "detect_communities")
        assert "detect_communities" in grapsit.__all__
        assert "CommunityConfigBuilder" in grapsit.__all__
