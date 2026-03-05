"""retrico CLI — command-line interface for the retrico Graph RAG package."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import click
import yaml


# ---------------------------------------------------------------------------
# Connection config file helpers
# ---------------------------------------------------------------------------

_CONFIG_FILE = ".retrico.yaml"


def _load_saved_config() -> Dict[str, Any]:
    """Load store config from .retrico.yaml in cwd (if it exists)."""
    p = Path(_CONFIG_FILE)
    if p.exists():
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        return data.get("store", {})
    return {}


def _save_config(store_cfg: Dict[str, Any]) -> None:
    """Save store config to .retrico.yaml."""
    with open(_CONFIG_FILE, "w") as f:
        yaml.dump({"store": store_cfg}, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Shared store option decorator
# ---------------------------------------------------------------------------

def _store_options(fn):
    """Attach common store connection flags to a click command."""
    opts = [
        click.option("--store-type", type=click.Choice(["falkordb_lite", "neo4j", "falkordb", "memgraph"]), default=None, help="Graph store backend."),
        click.option("--falkordb-lite-db-path", default=None, help="FalkorDBLite database file path."),
        click.option("--falkordb-lite-graph", default=None, help="FalkorDBLite graph name."),
        click.option("--neo4j-uri", default=None, help="Neo4j bolt URI."),
        click.option("--neo4j-user", default=None, help="Neo4j user."),
        click.option("--neo4j-password", default=None, help="Neo4j password."),
        click.option("--neo4j-database", default=None, help="Neo4j database name."),
        click.option("--falkordb-host", default=None, help="FalkorDB host."),
        click.option("--falkordb-port", default=None, type=int, help="FalkorDB port."),
        click.option("--falkordb-graph", default=None, help="FalkorDB graph name."),
        click.option("--memgraph-uri", default=None, help="Memgraph bolt URI."),
        click.option("--memgraph-user", default=None, help="Memgraph user."),
        click.option("--memgraph-password", default=None, help="Memgraph password."),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


def _resolve_store_kwargs(**cli_kwargs) -> Dict[str, Any]:
    """Merge saved config with explicit CLI flags. CLI flags win."""
    saved = _load_saved_config()
    # Build flat dict: start with saved, overlay CLI non-None values
    result = dict(saved)
    for cli_key, val in cli_kwargs.items():
        if val is not None:
            result[cli_key] = val
    # Ensure store_type has a default
    result.setdefault("store_type", "falkordb_lite")
    return result


def _make_store_config(kw: Dict[str, Any]):
    """Create a BaseStoreConfig from the merged kwargs dict."""
    from retrico.store.config import resolve_store_config
    return resolve_store_config(**kw)


def _open_store(kw: Dict[str, Any]):
    """Create and return a graph store from merged kwargs."""
    from retrico.store import create_graph_store
    cfg = _make_store_config(kw)
    store = create_graph_store(cfg)
    return store



# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

def _echo_success(msg: str):
    click.echo(click.style(msg, fg="green"))


def _echo_error(msg: str):
    click.echo(click.style(f"Error: {msg}", fg="red"), err=True)


def _echo_warning(msg: str):
    click.echo(click.style(f"Warning: {msg}", fg="yellow"), err=True)


def _echo_info(msg: str):
    click.echo(click.style(msg, fg="cyan"))


def _echo_table(rows: List[Dict[str, Any]], columns: List[str] = None):
    """Print a simple table from a list of dicts."""
    if not rows:
        click.echo("(no results)")
        return
    if columns is None:
        columns = list(rows[0].keys())
    # Compute column widths
    widths = {c: len(c) for c in columns}
    for row in rows:
        for c in columns:
            val = str(row.get(c, ""))
            widths[c] = max(widths[c], min(len(val), 60))
    # Header
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    click.echo(click.style(header, bold=True))
    click.echo("  ".join("-" * widths[c] for c in columns))
    # Rows
    for row in rows:
        line = "  ".join(str(row.get(c, ""))[:60].ljust(widths[c]) for c in columns)
        click.echo(line)


# ---------------------------------------------------------------------------
# Interactive wizard helpers
# ---------------------------------------------------------------------------

def _prompt_store_connection() -> Dict[str, Any]:
    """Interactively prompt for store connection details."""
    store_type = click.prompt(
        "Store type", type=click.Choice(["falkordb_lite", "neo4j", "falkordb", "memgraph"]),
        default="falkordb_lite",
    )
    cfg: Dict[str, Any] = {"store_type": store_type}
    if store_type == "falkordb_lite":
        cfg["falkordb_lite_db_path"] = click.prompt("Database file path", default="retrico.db")
        cfg["falkordb_lite_graph"] = click.prompt("Graph name", default="retrico")
    elif store_type == "neo4j":
        cfg["neo4j_uri"] = click.prompt("Neo4j URI", default="bolt://localhost:7687")
        cfg["neo4j_user"] = click.prompt("Neo4j user", default="neo4j")
        cfg["neo4j_password"] = click.prompt("Neo4j password", default="password", hide_input=True)
        cfg["neo4j_database"] = click.prompt("Neo4j database", default="neo4j")
    elif store_type == "falkordb":
        cfg["falkordb_host"] = click.prompt("FalkorDB host", default="localhost")
        cfg["falkordb_port"] = click.prompt("FalkorDB port", default=6379, type=int)
        cfg["falkordb_graph"] = click.prompt("FalkorDB graph", default="retrico")
    elif store_type == "memgraph":
        cfg["memgraph_uri"] = click.prompt("Memgraph URI", default="bolt://localhost:7687")
        cfg["memgraph_user"] = click.prompt("Memgraph user", default="")
        cfg["memgraph_password"] = click.prompt("Memgraph password", default="", hide_input=True)
    return cfg


def _prompt_labels(prompt_text: str) -> List[str]:
    """Prompt for comma-separated labels."""
    raw = click.prompt(prompt_text)
    return [l.strip() for l in raw.split(",") if l.strip()]


def _read_texts_interactive() -> List[str]:
    """Interactively get input texts."""
    source = click.prompt(
        "Input source",
        type=click.Choice(["text", "file"]),
        default="text",
    )
    if source == "text":
        click.echo("Enter texts (one per line, empty line to finish):")
        texts = []
        while True:
            line = click.prompt("", default="", show_default=False)
            if not line:
                break
            texts.append(line)
        if not texts:
            _echo_error("No texts provided.")
            sys.exit(1)
        return texts
    else:
        path = click.prompt("File path")
        p = Path(path)
        if not p.exists():
            _echo_error(f"File not found: {path}")
            sys.exit(1)
        if p.suffix == ".json":
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(item) for item in data]
            return [json.dumps(data)]
        return [p.read_text()]


# ---------------------------------------------------------------------------
# Main CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="retrico")
def cli():
    """retrico — End-to-end Graph RAG CLI.

    Build knowledge graphs, query them, detect communities, and manage
    graph data from the command line.
    """
    pass


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--show", is_flag=True, help="Show current saved connection.")
@click.option("--clear", is_flag=True, help="Clear saved connection.")
@_store_options
def connect(show, clear, **kwargs):
    """Save database connection to .retrico.yaml."""
    if show:
        saved = _load_saved_config()
        if not saved:
            click.echo("No saved connection. Run 'retrico connect' to set one up.")
        else:
            # Mask password
            display = dict(saved)
            if "password" in display and display["password"]:
                display["password"] = "****"
            if "neo4j_password" in display and display["neo4j_password"]:
                display["neo4j_password"] = "****"
            click.echo(yaml.dump({"store": display}, default_flow_style=False, sort_keys=False).strip())
        return

    if clear:
        p = Path(_CONFIG_FILE)
        if p.exists():
            p.unlink()
            _echo_success("Cleared saved connection.")
        else:
            click.echo("No saved connection to clear.")
        return

    # Check if any explicit flags were provided
    has_flags = any(v is not None for v in kwargs.values())
    if has_flags:
        store_kw = _resolve_store_kwargs(**kwargs)
    else:
        store_kw = _prompt_store_connection()

    _save_config(store_kw)
    _echo_success(f"Saved connection to {_CONFIG_FILE}")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--config", "config_file", type=click.Path(exists=True), help="YAML pipeline config file.")
@click.option("--text", "texts", multiple=True, help="Input text (repeatable).")
@click.option("--file", "files", multiple=True, type=click.Path(exists=True), help="Input text file (repeatable).")
@click.option("--entity-labels", default=None, help="Comma-separated entity labels.")
@click.option("--relation-labels", default=None, help="Comma-separated relation labels.")
@click.option("--method", type=click.Choice(["gliner", "llm"]), default=None, help="NER/relex method.")
@click.option("--chunk-method", default=None, help="Chunking method (sentence/paragraph/fixed).")
@click.option("--ner-model", default=None, help="NER model name.")
@click.option("--relex-model", default=None, help="Relex model name.")
@click.option("--api-key", default=None, envvar="OPENAI_API_KEY", help="OpenAI API key (or set OPENAI_API_KEY env var).")
@click.option("--llm-model", default=None, help="LLM model name.")
@click.option("--json-output", default=None, help="Save extracted data as JSON.")
@click.option("--embed-chunks", is_flag=True, help="Embed chunks.")
@click.option("--embed-entities", is_flag=True, help="Embed entities.")
@click.option("--verbose", is_flag=True, help="Verbose output.")
@click.option("--interactive", is_flag=True, help="Force interactive wizard.")
@click.option("--save-config", "save_config_file", default=None, help="Save pipeline config to YAML.")
@_store_options
def build(config_file, texts, files, entity_labels, relation_labels, method,
          chunk_method, ner_model, relex_model, api_key, llm_model,
          json_output, embed_chunks, embed_entities, verbose, interactive,
          save_config_file, **store_kwargs):
    """Build a knowledge graph from text."""
    import retrico

    store_kw = _resolve_store_kwargs(**store_kwargs)

    # Config file mode
    if config_file and not interactive:
        _echo_info(f"Loading pipeline from {config_file}")
        # Collect texts from flags/files
        all_texts = list(texts)
        for fp in files:
            all_texts.append(Path(fp).read_text())
        executor = retrico.ProcessorFactory.create_pipeline(config_file)
        input_data = {}
        if all_texts:
            input_data["texts"] = all_texts
        ctx = executor.run(input_data)
        _echo_success("Build complete.")
        if ctx.has("writer_result"):
            wr = ctx.get("writer_result")
            click.echo(f"  Entities: {wr.get('entity_count', 0)}")
            click.echo(f"  Relations: {wr.get('relation_count', 0)}")
            click.echo(f"  Chunks: {wr.get('chunk_count', 0)}")
        return

    # Check if we have enough for argument-based mode
    has_args = bool(texts or files) and entity_labels and not interactive
    if has_args:
        all_texts = list(texts)
        for fp in files:
            all_texts.append(Path(fp).read_text())
        ent_labels = [l.strip() for l in entity_labels.split(",")]
        rel_labels = [l.strip() for l in relation_labels.split(",")] if relation_labels else None

        build_kwargs: Dict[str, Any] = {
            "texts": all_texts,
            "entity_labels": ent_labels,
            "relation_labels": rel_labels,
            "verbose": verbose,
            "json_output": json_output,
            "embed_chunks": embed_chunks,
            "embed_entities": embed_entities,
            "store_config": _make_store_config(store_kw),
        }
        if chunk_method:
            build_kwargs["chunk_method"] = chunk_method
        if ner_model:
            build_kwargs["ner_model"] = ner_model
        if relex_model:
            build_kwargs["relex_model"] = relex_model
        # LLM method uses build_graph with LLM builder instead
        if method == "llm":
            _run_llm_build(all_texts, ent_labels, rel_labels, api_key, llm_model,
                           store_kw, verbose, json_output, embed_chunks, embed_entities,
                           chunk_method, save_config_file)
            return

        if save_config_file:
            builder = retrico.RetriCoBuilder(name="cli_build")
            builder.store(_make_store_config(store_kw))
            builder.chunker(method=chunk_method or "sentence")
            builder.ner_gliner(model=ner_model or "gliner-community/gliner_small-v2.5", labels=ent_labels)
            if rel_labels:
                builder.relex_gliner(
                    model=relex_model or "knowledgator/gliner-relex-large-v0.5",
                    entity_labels=ent_labels, relation_labels=rel_labels,
                )
            builder.graph_writer(json_output=json_output)
            builder.save(save_config_file)
            _echo_success(f"Config saved to {save_config_file}")

        ctx = retrico.build_graph(**build_kwargs)
        _echo_success("Build complete.")
        if ctx.has("writer_result"):
            wr = ctx.get("writer_result")
            click.echo(f"  Entities: {wr.get('entity_count', 0)}")
            click.echo(f"  Relations: {wr.get('relation_count', 0)}")
            click.echo(f"  Chunks: {wr.get('chunk_count', 0)}")
        return

    # Interactive wizard
    _echo_info("Build pipeline wizard")
    click.echo()

    # Step 1-2: Input
    all_texts = _read_texts_interactive()
    click.echo(f"  Got {len(all_texts)} text(s)")

    # Step 3-4: Store connection
    if not any(v is not None for v in store_kwargs.values()):
        saved = _load_saved_config()
        if saved:
            use_saved = click.confirm("Use saved connection?", default=True)
            if not use_saved:
                store_kw = _prompt_store_connection()
        else:
            store_kw = _prompt_store_connection()

    # Step 5: Chunking
    _chunk_method = click.prompt(
        "Chunking method", type=click.Choice(["sentence", "paragraph", "fixed"]),
        default=chunk_method or "sentence",
    )

    # Step 6: NER method
    _method = click.prompt(
        "NER method", type=click.Choice(["gliner", "llm"]),
        default=method or "gliner",
    )

    # Step 7: Entity labels
    _ent_labels = _prompt_labels("Entity labels (comma-separated)")

    # Step 8: Relation extraction
    _rel_labels = None
    if click.confirm("Add relation extraction?", default=True):
        _rel_labels = _prompt_labels("Relation labels (comma-separated)")

    # Step 9: LLM config
    _api_key = api_key
    _llm_model_name = llm_model
    if _method == "llm":
        if not _api_key:
            _api_key = click.prompt("API key", hide_input=True)
        if not _llm_model_name:
            _llm_model_name = click.prompt("LLM model", default="gpt-4o-mini")

    # Step 10: Embeddings
    _embed_chunks = embed_chunks or click.confirm("Embed chunks?", default=False)
    _embed_entities = embed_entities or click.confirm("Embed entities?", default=False)

    # Step 11: Save config
    _save_path = save_config_file
    if not _save_path and click.confirm("Save config for reuse?", default=False):
        _save_path = click.prompt("Config output path", default="build_config.yaml")

    click.echo()
    _echo_info("Running pipeline...")

    if _method == "llm":
        _run_llm_build(all_texts, _ent_labels, _rel_labels, _api_key, _llm_model_name,
                        store_kw, verbose, json_output, _embed_chunks, _embed_entities,
                        _chunk_method, _save_path)
    else:
        build_kwargs = {
            "texts": all_texts,
            "entity_labels": _ent_labels,
            "relation_labels": _rel_labels,
            "chunk_method": _chunk_method,
            "verbose": verbose,
            "json_output": json_output,
            "embed_chunks": _embed_chunks,
            "embed_entities": _embed_entities,
            "store_config": _make_store_config(store_kw),
        }
        if ner_model:
            build_kwargs["ner_model"] = ner_model
        if relex_model:
            build_kwargs["relex_model"] = relex_model

        if _save_path:
            builder = retrico.RetriCoBuilder(name="cli_build")
            builder.store(_make_store_config(store_kw))
            builder.chunker(method=_chunk_method)
            builder.ner_gliner(model=ner_model or "gliner-community/gliner_small-v2.5", labels=_ent_labels)
            if _rel_labels:
                builder.relex_gliner(
                    model=relex_model or "knowledgator/gliner-relex-large-v0.5",
                    entity_labels=_ent_labels, relation_labels=_rel_labels,
                )
            builder.graph_writer(json_output=json_output)
            builder.save(_save_path)
            _echo_success(f"Config saved to {_save_path}")

        ctx = retrico.build_graph(**build_kwargs)
        _echo_success("Build complete.")
        if ctx.has("writer_result"):
            wr = ctx.get("writer_result")
            click.echo(f"  Entities: {wr.get('entity_count', 0)}")
            click.echo(f"  Relations: {wr.get('relation_count', 0)}")
            click.echo(f"  Chunks: {wr.get('chunk_count', 0)}")


def _run_llm_build(texts, ent_labels, rel_labels, api_key, llm_model,
                    store_kw, verbose, json_output, embed_chunks, embed_entities,
                    chunk_method, save_config_file):
    """Run a build pipeline using LLM-based NER/relex."""
    import retrico

    builder = retrico.RetriCoBuilder(name="cli_build_llm")
    builder.store(_make_store_config(store_kw))
    builder.chunker(method=chunk_method or "sentence")
    builder.ner_llm(api_key=api_key, model=llm_model or "gpt-4o-mini", labels=ent_labels)
    if rel_labels:
        builder.relex_llm(
            api_key=api_key, model=llm_model or "gpt-4o-mini",
            entity_labels=ent_labels, relation_labels=rel_labels,
        )
    builder.graph_writer(json_output=json_output)

    if save_config_file:
        builder.save(save_config_file)
        _echo_success(f"Config saved to {save_config_file}")

    executor = builder.build(verbose=verbose)
    ctx = executor.run(texts=texts)
    _echo_success("Build complete.")
    if ctx.has("writer_result"):
        wr = ctx.get("writer_result")
        click.echo(f"  Entities: {wr.get('entity_count', 0)}")
        click.echo(f"  Relations: {wr.get('relation_count', 0)}")
        click.echo(f"  Chunks: {wr.get('chunk_count', 0)}")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--json-output", default=None, help="Save data as JSON.")
@click.option("--verbose", is_flag=True, help="Verbose output.")
@_store_options
def ingest(file, json_output, verbose, **store_kwargs):
    """Ingest structured JSON data into the graph."""
    import retrico

    store_kw = _resolve_store_kwargs(**store_kwargs)

    with open(file) as f:
        data = json.load(f)

    if not isinstance(data, list):
        _echo_error("JSON file must contain a list of objects.")
        sys.exit(1)

    ctx = retrico.ingest_data(
        data=data,
        json_output=json_output,
        verbose=verbose,
        store_config=_make_store_config(store_kw),
    )
    _echo_success("Ingest complete.")
    if ctx.has("writer_result"):
        wr = ctx.get("writer_result")
        click.echo(f"  Entities: {wr.get('entity_count', 0)}")
        click.echo(f"  Relations: {wr.get('relation_count', 0)}")
        click.echo(f"  Chunks: {wr.get('chunk_count', 0)}")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("query_text", required=False)
@click.option("--config", "config_file", type=click.Path(exists=True), help="YAML pipeline config.")
@click.option("--entity-labels", default=None, help="Comma-separated entity labels.")
@click.option("--strategy", default=None, help="Retrieval strategy (entity/community/path/chunk_embedding/entity_embedding/tool/keyword). Comma-separated for multi.")
@click.option("--method", type=click.Choice(["gliner", "llm"]), default=None, help="NER method for query parsing.")
@click.option("--api-key", default=None, envvar="OPENAI_API_KEY", help="OpenAI API key (or set OPENAI_API_KEY env var).")
@click.option("--llm-model", default=None, help="LLM model name.")
@click.option("--max-hops", default=None, type=int, help="Subgraph expansion depth.")
@click.option("--verbose", is_flag=True, help="Verbose output.")
@click.option("--interactive", is_flag=True, help="Force interactive wizard.")
@_store_options
def query(query_text, config_file, entity_labels, strategy, method, api_key,
          llm_model, max_hops, verbose, interactive, **store_kwargs):
    """Query the knowledge graph."""
    import retrico

    store_kw = _resolve_store_kwargs(**store_kwargs)

    # Config file mode
    if config_file and not interactive:
        if not query_text:
            query_text = click.prompt("Query")
        executor = retrico.ProcessorFactory.create_pipeline(config_file)
        ctx = executor.run(query=query_text)
        _print_query_result(ctx, query_text)
        return

    # Check if we have enough for argument-based mode
    has_args = query_text and entity_labels and not interactive
    if has_args:
        ent_labels = [l.strip() for l in entity_labels.split(",")]
        query_kwargs: Dict[str, Any] = {
            "query": query_text,
            "entity_labels": ent_labels,
            "verbose": verbose,
            "store_config": _make_store_config(store_kw),
        }
        if method:
            query_kwargs["ner_method"] = method
        if api_key:
            query_kwargs["api_key"] = api_key
        if llm_model:
            query_kwargs["model"] = llm_model
        if max_hops is not None:
            query_kwargs["max_hops"] = max_hops
        if strategy:
            strategies = [s.strip() for s in strategy.split(",")]
            if len(strategies) == 1:
                query_kwargs["retrieval_strategy"] = strategies[0]
            else:
                query_kwargs["retrieval_strategy"] = strategies

        result = retrico.query_graph(**query_kwargs)
        _print_query_result_obj(result)
        return

    # Interactive wizard
    _echo_info("Query pipeline wizard")
    click.echo()

    if not query_text:
        query_text = click.prompt("Enter your query")

    # Store connection
    if not any(v is not None for v in store_kwargs.values()):
        saved = _load_saved_config()
        if saved:
            use_saved = click.confirm("Use saved connection?", default=True)
            if not use_saved:
                store_kw = _prompt_store_connection()
        else:
            store_kw = _prompt_store_connection()

    # Strategy
    _strategy = click.prompt(
        "Retrieval strategy",
        type=click.Choice(["entity", "community", "path", "chunk_embedding", "entity_embedding", "tool", "keyword"]),
        default=strategy or "entity",
    )

    # NER method + labels
    _method = click.prompt("NER method", type=click.Choice(["gliner", "llm"]), default=method or "gliner")
    _ent_labels = _prompt_labels("Entity labels (comma-separated)")

    # LLM config
    _api_key = api_key
    _llm_model_name = llm_model
    if click.confirm("Add LLM reasoner?", default=bool(api_key)):
        if not _api_key:
            _api_key = click.prompt("API key", hide_input=True)
        if not _llm_model_name:
            _llm_model_name = click.prompt("LLM model", default="gpt-4o-mini")

    click.echo()
    _echo_info("Running query...")

    query_kwargs = {
        "query": query_text,
        "entity_labels": _ent_labels,
        "ner_method": _method,
        "verbose": verbose,
        "store_config": _make_store_config(store_kw),
        "retrieval_strategy": _strategy,
    }
    if _api_key:
        query_kwargs["api_key"] = _api_key
    if _llm_model_name:
        query_kwargs["model"] = _llm_model_name
    if max_hops is not None:
        query_kwargs["max_hops"] = max_hops

    result = retrico.query_graph(**query_kwargs)
    _print_query_result_obj(result)


def _print_query_result(ctx, query_text: str):
    """Print query result from a PipeContext."""
    if ctx.has("reasoner_result"):
        result = ctx.get("reasoner_result")["result"]
        _print_query_result_obj(result)
    elif ctx.has("chunk_result"):
        from retrico.models import QueryResult
        subgraph = ctx.get("chunk_result")["subgraph"]
        result = QueryResult(query=query_text, subgraph=subgraph)
        _print_query_result_obj(result)
    else:
        _echo_warning("No results found.")


def _print_query_result_obj(result):
    """Print a QueryResult object."""
    if result.answer:
        click.echo()
        click.echo(click.style("Answer:", bold=True))
        click.echo(result.answer)
        click.echo()

    sg = result.subgraph
    if sg and sg.entities:
        click.echo(click.style(f"Entities ({len(sg.entities)}):", bold=True))
        for e in sg.entities[:20]:
            etype = f" [{e.entity_type}]" if e.entity_type else ""
            click.echo(f"  - {e.label}{etype} (id: {e.id[:8]}...)")
        if len(sg.entities) > 20:
            click.echo(f"  ... and {len(sg.entities) - 20} more")
        click.echo()

    if sg and sg.relations:
        click.echo(click.style(f"Relations ({len(sg.relations)}):", bold=True))
        for r in sg.relations[:20]:
            click.echo(f"  - {r.head_entity} --[{r.relation_type}]--> {r.tail_entity}")
        if len(sg.relations) > 20:
            click.echo(f"  ... and {len(sg.relations) - 20} more")
        click.echo()

    if sg and sg.chunks:
        click.echo(click.style(f"Source chunks ({len(sg.chunks)}):", bold=True))
        for c in sg.chunks[:5]:
            text_preview = c.text[:120] + "..." if len(c.text) > 120 else c.text
            click.echo(f"  - [{c.id[:8]}...] {text_preview}")
        if len(sg.chunks) > 5:
            click.echo(f"  ... and {len(sg.chunks) - 5} more")


# ---------------------------------------------------------------------------
# community
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--config", "config_file", type=click.Path(exists=True), help="YAML pipeline config.")
@click.option("--method", type=click.Choice(["louvain", "leiden"]), default=None, help="Community detection method.")
@click.option("--levels", default=None, type=int, help="Hierarchical levels.")
@click.option("--resolution", default=None, type=float, help="Resolution parameter.")
@click.option("--api-key", default=None, envvar="OPENAI_API_KEY", help="OpenAI API key (or set OPENAI_API_KEY env var).")
@click.option("--llm-model", default=None, help="LLM model for summarization.")
@click.option("--verbose", is_flag=True, help="Verbose output.")
@click.option("--interactive", is_flag=True, help="Force interactive wizard.")
@_store_options
def community(config_file, method, levels, resolution, api_key, llm_model,
              verbose, interactive, **store_kwargs):
    """Detect communities in the knowledge graph."""
    import retrico

    store_kw = _resolve_store_kwargs(**store_kwargs)

    if config_file and not interactive:
        executor = retrico.ProcessorFactory.create_pipeline(config_file)
        ctx = executor.run()
        _echo_success("Community detection complete.")
        if ctx.has("detector_result"):
            dr = ctx.get("detector_result")
            click.echo(f"  Communities: {dr.get('num_communities', '?')}")
        return

    # Check for argument-based mode
    has_args = not interactive and any(v is not None for v in [method, levels, resolution])
    if has_args or not interactive and _load_saved_config():
        comm_kwargs: Dict[str, Any] = {
            "verbose": verbose,
            "store_config": _make_store_config(store_kw),
        }
        if method:
            comm_kwargs["method"] = method
        if levels is not None:
            comm_kwargs["levels"] = levels
        if resolution is not None:
            comm_kwargs["resolution"] = resolution
        if api_key:
            comm_kwargs["api_key"] = api_key
        if llm_model:
            comm_kwargs["model"] = llm_model

        ctx = retrico.detect_communities(**comm_kwargs)
        _echo_success("Community detection complete.")
        if ctx.has("detector_result"):
            dr = ctx.get("detector_result")
            click.echo(f"  Communities: {dr.get('num_communities', '?')}")
        return

    # Interactive wizard
    _echo_info("Community detection wizard")
    click.echo()

    if not any(v is not None for v in store_kwargs.values()):
        saved = _load_saved_config()
        if saved:
            use_saved = click.confirm("Use saved connection?", default=True)
            if not use_saved:
                store_kw = _prompt_store_connection()
        else:
            store_kw = _prompt_store_connection()

    _method = click.prompt("Method", type=click.Choice(["louvain", "leiden"]), default="louvain")
    _levels = click.prompt("Levels", default=1, type=int)
    _resolution = click.prompt("Resolution", default=1.0, type=float)

    _api_key = api_key
    if click.confirm("Add LLM summarization?", default=False):
        if not _api_key:
            _api_key = click.prompt("API key", hide_input=True)

    click.echo()
    _echo_info("Running community detection...")

    comm_kwargs = {
        "method": _method,
        "levels": _levels,
        "resolution": _resolution,
        "verbose": verbose,
        "store_config": _make_store_config(store_kw),
    }
    if _api_key:
        comm_kwargs["api_key"] = _api_key
        if llm_model:
            comm_kwargs["model"] = llm_model

    ctx = retrico.detect_communities(**comm_kwargs)
    _echo_success("Community detection complete.")
    if ctx.has("detector_result"):
        dr = ctx.get("detector_result")
        click.echo(f"  Communities: {dr.get('num_communities', '?')}")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--config", "config_file", type=click.Path(exists=True), help="YAML pipeline config.")
@click.option("--kg-model", default=None, help="PyKEEN model (RotatE/TransE/ComplEx).")
@click.option("--embedding-dim", default=None, type=int, help="Embedding dimension.")
@click.option("--epochs", default=None, type=int, help="Training epochs.")
@click.option("--batch-size", default=None, type=int, help="Batch size.")
@click.option("--lr", default=None, type=float, help="Learning rate.")
@click.option("--device", default=None, help="Device (cpu/cuda).")
@click.option("--model-path", default=None, help="Path to save model.")
@click.option("--verbose", is_flag=True, help="Verbose output.")
@click.option("--interactive", is_flag=True, help="Force interactive wizard.")
@_store_options
def model(config_file, kg_model, embedding_dim, epochs, batch_size, lr,
          device, model_path, verbose, interactive, **store_kwargs):
    """Train KG embeddings."""
    import retrico

    store_kw = _resolve_store_kwargs(**store_kwargs)

    if config_file and not interactive:
        executor = retrico.ProcessorFactory.create_pipeline(config_file)
        ctx = executor.run()
        _echo_success("KG model training complete.")
        return

    # Check for argument-based mode
    has_args = not interactive and any(v is not None for v in [kg_model, epochs, embedding_dim])
    if has_args:
        model_kwargs: Dict[str, Any] = {
            "verbose": verbose,
            "store_config": _make_store_config(store_kw),
        }
        if kg_model:
            model_kwargs["model"] = kg_model
        if embedding_dim is not None:
            model_kwargs["embedding_dim"] = embedding_dim
        if epochs is not None:
            model_kwargs["epochs"] = epochs
        if batch_size is not None:
            model_kwargs["batch_size"] = batch_size
        if lr is not None:
            model_kwargs["lr"] = lr
        if device:
            model_kwargs["device"] = device
        if model_path:
            model_kwargs["model_path"] = model_path

        ctx = retrico.train_kg_model(**model_kwargs)
        _echo_success("KG model training complete.")
        if ctx.has("trainer_result"):
            tr = ctx.get("trainer_result")
            click.echo(f"  Model: {tr.get('model_name', '?')}")
            if "metrics" in tr:
                for k, v in tr["metrics"].items():
                    click.echo(f"  {k}: {v}")
        return

    # Interactive wizard
    _echo_info("KG model training wizard")
    click.echo()

    if not any(v is not None for v in store_kwargs.values()):
        saved = _load_saved_config()
        if saved:
            use_saved = click.confirm("Use saved connection?", default=True)
            if not use_saved:
                store_kw = _prompt_store_connection()
        else:
            store_kw = _prompt_store_connection()

    _kg_model = click.prompt("Model", type=click.Choice(["RotatE", "TransE", "ComplEx"]), default=kg_model or "RotatE")
    _embedding_dim = click.prompt("Embedding dim", default=embedding_dim or 128, type=int)
    _epochs = click.prompt("Epochs", default=epochs or 100, type=int)
    _batch_size = click.prompt("Batch size", default=batch_size or 256, type=int)
    _lr = click.prompt("Learning rate", default=lr or 0.001, type=float)
    _device = click.prompt("Device", default=device or "cpu")
    _model_path = click.prompt("Model save path", default=model_path or "kg_model")

    click.echo()
    _echo_info("Training KG model...")

    ctx = retrico.train_kg_model(
        model=_kg_model,
        embedding_dim=_embedding_dim,
        epochs=_epochs,
        batch_size=_batch_size,
        lr=_lr,
        device=_device,
        model_path=_model_path,
        verbose=verbose,
        store_config=_make_store_config(store_kw),
    )
    _echo_success("KG model training complete.")
    if ctx.has("trainer_result"):
        tr = ctx.get("trainer_result")
        click.echo(f"  Model: {tr.get('model_name', '?')}")


# ---------------------------------------------------------------------------
# init — generate YAML config interactively
# ---------------------------------------------------------------------------

@cli.command("init")
@click.argument("pipeline_type", required=False, type=click.Choice(["build", "query", "community", "model"]))
def init_config(pipeline_type):
    """Generate a pipeline config YAML interactively."""
    import retrico

    if not pipeline_type:
        pipeline_type = click.prompt(
            "Pipeline type",
            type=click.Choice(["build", "query", "community", "model"]),
        )

    # Store connection
    store_kw = _prompt_store_connection()

    if pipeline_type == "build":
        builder = retrico.RetriCoBuilder(name="generated_build")
        builder.store(_make_store_config(store_kw))

        chunk_method = click.prompt("Chunking method", type=click.Choice(["sentence", "paragraph", "fixed"]), default="sentence")
        builder.chunker(method=chunk_method)

        ner_method = click.prompt("NER method", type=click.Choice(["gliner", "llm"]), default="gliner")
        ent_labels = _prompt_labels("Entity labels (comma-separated)")

        if ner_method == "gliner":
            ner_model = click.prompt("GLiNER NER model", default="gliner-community/gliner_small-v2.5")
            builder.ner_gliner(model=ner_model, labels=ent_labels)
        else:
            api_key = os.environ.get("OPENAI_API_KEY") or click.prompt("API key (or set OPENAI_API_KEY)", hide_input=True)
            llm_model = click.prompt("LLM model", default="gpt-4o-mini")
            builder.ner_llm(api_key=api_key, model=llm_model, labels=ent_labels)

        if click.confirm("Add relation extraction?", default=True):
            rel_labels = _prompt_labels("Relation labels (comma-separated)")
            if ner_method == "gliner":
                relex_model = click.prompt("GLiNER relex model", default="knowledgator/gliner-relex-large-v0.5")
                builder.relex_gliner(model=relex_model, entity_labels=ent_labels, relation_labels=rel_labels)
            else:
                builder.relex_llm(api_key=api_key, model=llm_model, entity_labels=ent_labels, relation_labels=rel_labels)

        builder.graph_writer()
        output = click.prompt("Output file", default="build_config.yaml")
        builder.save(output)

    elif pipeline_type == "query":
        builder = retrico.RetriCoSearch(name="generated_query")
        builder.store(_make_store_config(store_kw))

        ner_method = click.prompt("NER method", type=click.Choice(["gliner", "llm"]), default="gliner")
        ent_labels = _prompt_labels("Entity labels (comma-separated)")

        parser_kw: Dict[str, Any] = {"method": ner_method, "labels": ent_labels}
        if ner_method == "llm":
            api_key = os.environ.get("OPENAI_API_KEY") or click.prompt("API key (or set OPENAI_API_KEY)", hide_input=True)
            llm_model = click.prompt("LLM model", default="gpt-4o-mini")
            parser_kw["api_key"] = api_key
            parser_kw["model"] = llm_model
        builder.query_parser(**parser_kw)

        strategy = click.prompt(
            "Retrieval strategy",
            type=click.Choice(["entity", "community", "path", "chunk_embedding", "entity_embedding"]),
            default="entity",
        )
        if strategy == "entity":
            max_hops = click.prompt("Max hops", default=2, type=int)
            builder.retriever(max_hops=max_hops)
        elif strategy == "community":
            builder.community_retriever()
        elif strategy == "path":
            builder.path_retriever()
        elif strategy == "chunk_embedding":
            builder.chunk_embedding_retriever()
        elif strategy == "entity_embedding":
            builder.entity_embedding_retriever()

        builder.chunk_retriever()

        if click.confirm("Add LLM reasoner?", default=False):
            if ner_method != "llm":
                api_key = os.environ.get("OPENAI_API_KEY") or click.prompt("API key (or set OPENAI_API_KEY)", hide_input=True)
                llm_model = click.prompt("LLM model", default="gpt-4o-mini")
            builder.reasoner(api_key=api_key, model=llm_model)

        output = click.prompt("Output file", default="query_config.yaml")
        builder.save(output)

    elif pipeline_type == "community":
        builder = retrico.RetriCoCommunity(name="generated_community")
        builder.store(_make_store_config(store_kw))

        method = click.prompt("Method", type=click.Choice(["louvain", "leiden"]), default="louvain")
        levels = click.prompt("Levels", default=1, type=int)
        resolution = click.prompt("Resolution", default=1.0, type=float)
        builder.detector(method=method, levels=levels, resolution=resolution)

        if click.confirm("Add LLM summarization?", default=False):
            api_key = os.environ.get("OPENAI_API_KEY") or click.prompt("API key (or set OPENAI_API_KEY)", hide_input=True)
            llm_model = click.prompt("LLM model", default="gpt-4o-mini")
            builder.summarizer(api_key=api_key, model=llm_model)
            builder.embedder()

        output = click.prompt("Output file", default="community_config.yaml")
        builder.save(output)

    elif pipeline_type == "model":
        builder = retrico.RetriCoModeling(name="generated_model")
        builder.store(_make_store_config(store_kw))

        builder.triple_reader()
        kg_model = click.prompt("Model", type=click.Choice(["RotatE", "TransE", "ComplEx"]), default="RotatE")
        embedding_dim = click.prompt("Embedding dim", default=128, type=int)
        epochs = click.prompt("Epochs", default=100, type=int)
        builder.trainer(model=kg_model, embedding_dim=embedding_dim, epochs=epochs)
        model_path = click.prompt("Model save path", default="kg_model")
        builder.storer(model_path=model_path)

        output = click.prompt("Output file", default="model_config.yaml")
        builder.save(output)

    _echo_success(f"Config saved to {output}")


# ---------------------------------------------------------------------------
# graph — direct CRUD subcommands
# ---------------------------------------------------------------------------

@cli.group()
def graph():
    """Direct graph database operations."""
    pass


@graph.command("entities")
@click.option("--type", "entity_type", default=None, help="Filter by entity type.")
@click.option("--limit", default=50, type=int, help="Max results.")
@_store_options
def graph_entities(entity_type, limit, **store_kwargs):
    """List entities in the graph."""
    store_kw = _resolve_store_kwargs(**store_kwargs)
    store = _open_store(store_kw)
    try:
        entities = store.get_all_entities()
        if entity_type:
            entities = [e for e in entities if e.get("entity_type", "").lower() == entity_type.lower()]
        entities = entities[:limit]
        if not entities:
            click.echo("No entities found.")
            return
        _echo_table(entities, columns=["id", "label", "entity_type"])
    finally:
        store.close()


@graph.command("relations")
@click.argument("entity")
@_store_options
def graph_relations(entity, **store_kwargs):
    """List relations for an entity (by label or ID)."""
    store_kw = _resolve_store_kwargs(**store_kwargs)
    store = _open_store(store_kw)
    try:
        # Try by label first, then by ID
        ent = store.get_entity_by_label(entity)
        if not ent:
            ent = store.get_entity_by_id(entity)
        if not ent:
            _echo_error(f"Entity not found: {entity}")
            return
        relations = store.get_entity_relations(ent["id"])
        if not relations:
            click.echo("No relations found.")
            return
        _echo_table(relations)
    finally:
        store.close()


@graph.command("search")
@click.argument("query_text")
@click.option("--top-k", default=10, type=int, help="Max results.")
@_store_options
def graph_search(query_text, top_k, **store_kwargs):
    """Full-text search chunks."""
    store_kw = _resolve_store_kwargs(**store_kwargs)
    store = _open_store(store_kw)
    try:
        results = store.fulltext_search_chunks(query_text, top_k=top_k)
        if not results:
            click.echo("No results found.")
            return
        for r in results:
            score = f" (score: {r['score']:.3f})" if "score" in r else ""
            text = r.get("text", "")
            text_preview = text[:150] + "..." if len(text) > 150 else text
            click.echo(f"  [{r.get('id', '?')[:8]}...]{score} {text_preview}")
    finally:
        store.close()


@graph.command("add-entity")
@click.argument("label")
@click.option("--type", "entity_type", required=True, help="Entity type.")
@click.option("--properties", default=None, help="JSON properties string.")
@_store_options
def graph_add_entity(label, entity_type, properties, **store_kwargs):
    """Add an entity to the graph."""
    store_kw = _resolve_store_kwargs(**store_kwargs)
    store = _open_store(store_kw)
    try:
        props = json.loads(properties) if properties else None
        entity_id = store.add_entity(label, entity_type, properties=props)
        _echo_success(f"Created entity: {label} (id: {entity_id})")
    finally:
        store.close()


@graph.command("add-relation")
@click.argument("head_label")
@click.argument("tail_label")
@click.argument("rel_type")
@_store_options
def graph_add_relation(head_label, tail_label, rel_type, **store_kwargs):
    """Add a relation between two entities (by label)."""
    store_kw = _resolve_store_kwargs(**store_kwargs)
    store = _open_store(store_kw)
    try:
        head = store.get_entity_by_label(head_label)
        if not head:
            _echo_error(f"Head entity not found: {head_label}")
            return
        tail = store.get_entity_by_label(tail_label)
        if not tail:
            _echo_error(f"Tail entity not found: {tail_label}")
            return
        rel_id = store.add_relation(head["id"], tail["id"], rel_type)
        _echo_success(f"Created relation: {head_label} --[{rel_type}]--> {tail_label} (id: {rel_id})")
    finally:
        store.close()


@graph.command("update")
@click.argument("entity_id")
@click.option("--label", default=None, help="New label.")
@click.option("--properties", default=None, help="JSON properties to merge.")
@_store_options
def graph_update(entity_id, label, properties, **store_kwargs):
    """Update an entity."""
    store_kw = _resolve_store_kwargs(**store_kwargs)
    store = _open_store(store_kw)
    try:
        props = json.loads(properties) if properties else None
        ok = store.update_entity(entity_id, label=label, properties=props)
        if ok:
            _echo_success(f"Updated entity {entity_id}")
        else:
            _echo_error(f"Entity not found: {entity_id}")
    finally:
        store.close()


@graph.command("delete")
@click.option("--entity", "entity_id", default=None, help="Entity ID to delete.")
@click.option("--relation", "relation_id", default=None, help="Relation ID to delete.")
@_store_options
def graph_delete(entity_id, relation_id, **store_kwargs):
    """Delete an entity or relation."""
    if not entity_id and not relation_id:
        _echo_error("Provide --entity or --relation ID.")
        return
    store_kw = _resolve_store_kwargs(**store_kwargs)
    store = _open_store(store_kw)
    try:
        if entity_id:
            ok = store.delete_entity(entity_id)
            if ok:
                _echo_success(f"Deleted entity {entity_id}")
            else:
                _echo_error(f"Entity not found: {entity_id}")
        if relation_id:
            ok = store.delete_relation(relation_id)
            if ok:
                _echo_success(f"Deleted relation {relation_id}")
            else:
                _echo_error(f"Relation not found: {relation_id}")
    finally:
        store.close()


@graph.command("merge")
@click.argument("source_id")
@click.argument("target_id")
@_store_options
def graph_merge(source_id, target_id, **store_kwargs):
    """Merge source entity into target entity."""
    store_kw = _resolve_store_kwargs(**store_kwargs)
    store = _open_store(store_kw)
    try:
        ok = store.merge_entities(source_id, target_id)
        if ok:
            _echo_success(f"Merged {source_id} into {target_id}")
        else:
            _echo_error("Merge failed — one or both entities not found.")
    finally:
        store.close()


@graph.command("stats")
@_store_options
def graph_stats(**store_kwargs):
    """Show graph statistics."""
    store_kw = _resolve_store_kwargs(**store_kwargs)
    store = _open_store(store_kw)
    try:
        # Count entities
        entities = store.get_all_entities()
        entity_count = len(entities)

        # Count by type
        type_counts: Dict[str, int] = {}
        for e in entities:
            etype = e.get("entity_type", "unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1

        click.echo(click.style("Graph Statistics", bold=True))
        click.echo(f"  Total entities: {entity_count}")
        if type_counts:
            click.echo("  By type:")
            for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
                click.echo(f"    {t}: {c}")

        # Try to get communities
        try:
            communities = store.get_all_communities()
            click.echo(f"  Communities: {len(communities)}")
        except (NotImplementedError, Exception):
            pass
    finally:
        store.close()


@graph.command("cypher")
@click.argument("query_text")
@_store_options
def graph_cypher(query_text, **store_kwargs):
    """Run a raw Cypher query."""
    store_kw = _resolve_store_kwargs(**store_kwargs)
    store = _open_store(store_kw)
    try:
        results = store.run_cypher(query_text)
        if not results:
            click.echo("(no results)")
            return
        # Try to display as table if results are dicts
        if isinstance(results[0], dict):
            _echo_table(results)
        else:
            for r in results:
                click.echo(r)
    finally:
        store.close()


@graph.command("clear")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
@_store_options
def graph_clear(yes, **store_kwargs):
    """Clear all data from the graph."""
    if not yes:
        click.confirm("This will delete ALL data. Continue?", abort=True)
    store_kw = _resolve_store_kwargs(**store_kwargs)
    store = _open_store(store_kw)
    try:
        store.clear_all()
        _echo_success("All data cleared.")
    finally:
        store.close()


# ---------------------------------------------------------------------------
# shell — interactive REPL
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--entity-labels", default=None, help="Default entity labels for queries.")
@click.option("--api-key", default=None, envvar="OPENAI_API_KEY", help="OpenAI API key (or set OPENAI_API_KEY env var).")
@click.option("--llm-model", default=None, help="LLM model name.")
@_store_options
def shell(entity_labels, api_key, llm_model, **store_kwargs):
    """Interactive query REPL."""
    store_kw = _resolve_store_kwargs(**store_kwargs)

    _echo_info("retrico interactive shell")
    click.echo("Type a query or :help for commands. :quit to exit.")
    click.echo()

    ent_labels = [l.strip() for l in entity_labels.split(",")] if entity_labels else None
    store = None

    def _ensure_store():
        nonlocal store
        if store is None:
            store = _open_store(store_kw)
        return store

    try:
        while True:
            try:
                line = click.prompt("retrico", prompt_suffix="> ")
            except (EOFError, KeyboardInterrupt):
                click.echo()
                break

            line = line.strip()
            if not line:
                continue

            if line in (":quit", ":exit", ":q"):
                break
            elif line == ":help":
                click.echo("Commands:")
                click.echo("  :entities [type]    — List entities")
                click.echo("  :relations ENTITY   — Show relations for entity")
                click.echo("  :search TEXT        — Full-text search chunks")
                click.echo("  :cypher QUERY       — Run raw Cypher")
                click.echo("  :labels TEXT        — Set default entity labels")
                click.echo("  :help               — Show this help")
                click.echo("  :quit               — Exit")
                click.echo("  (anything else)     — Run as query_graph()")
            elif line.startswith(":entities"):
                s = _ensure_store()
                parts = line.split(maxsplit=1)
                entities = s.get_all_entities()
                if len(parts) > 1:
                    etype = parts[1].strip()
                    entities = [e for e in entities if e.get("entity_type", "").lower() == etype.lower()]
                entities = entities[:50]
                _echo_table(entities, columns=["id", "label", "entity_type"])
            elif line.startswith(":relations"):
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    _echo_error("Usage: :relations ENTITY_LABEL")
                    continue
                s = _ensure_store()
                ent = s.get_entity_by_label(parts[1].strip())
                if not ent:
                    ent = s.get_entity_by_id(parts[1].strip())
                if not ent:
                    _echo_error(f"Entity not found: {parts[1]}")
                    continue
                rels = s.get_entity_relations(ent["id"])
                _echo_table(rels) if rels else click.echo("No relations.")
            elif line.startswith(":search"):
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    _echo_error("Usage: :search TEXT")
                    continue
                s = _ensure_store()
                results = s.fulltext_search_chunks(parts[1].strip())
                for r in results:
                    text = r.get("text", "")[:150]
                    click.echo(f"  [{r.get('id', '?')[:8]}] {text}")
                if not results:
                    click.echo("No results.")
            elif line.startswith(":cypher"):
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    _echo_error("Usage: :cypher QUERY")
                    continue
                s = _ensure_store()
                try:
                    results = s.run_cypher(parts[1].strip())
                    if results and isinstance(results[0], dict):
                        _echo_table(results)
                    elif results:
                        for r in results:
                            click.echo(r)
                    else:
                        click.echo("(no results)")
                except Exception as e:
                    _echo_error(str(e))
            elif line.startswith(":labels"):
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    if ent_labels:
                        click.echo(f"Current labels: {', '.join(ent_labels)}")
                    else:
                        click.echo("No labels set. Usage: :labels person,org,location")
                    continue
                ent_labels = [l.strip() for l in parts[1].split(",") if l.strip()]
                click.echo(f"Labels set: {', '.join(ent_labels)}")
            else:
                # Treat as a query
                if not ent_labels:
                    _echo_warning("No entity labels set. Use :labels to set them, or --entity-labels flag.")
                    continue
                try:
                    import retrico
                    query_kwargs: Dict[str, Any] = {
                        "query": line,
                        "entity_labels": ent_labels,
                        "store_config": _make_store_config(store_kw),
                    }
                    if api_key:
                        query_kwargs["api_key"] = api_key
                    if llm_model:
                        query_kwargs["model"] = llm_model
                    result = retrico.query_graph(**query_kwargs)
                    _print_query_result_obj(result)
                except Exception as e:
                    _echo_error(str(e))
    finally:
        if store is not None:
            store.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
