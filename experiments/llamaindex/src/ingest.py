"""
Ingestion stage (Postgres + pgvector edition).

Writes to the SAME database as the Java app (`ai_rag_assistant`) but a DIFFERENT
table (`data_llamaindex_chunks` — LlamaIndex prepends `data_` to the table name
you pass in). Your Java `document_chunks` table is untouched.

Two entry points:

    build_index(data_dir)   -> runs ingestion (reads files, embeds, writes to pg)
    load_index()            -> reconnects to the existing pg table, NO embedding

Call build_index once with --ingest. Every subsequent --mode run calls load_index
and goes straight to retrieval, so your timing numbers reflect retrieval + answer,
not re-embedding 500 chunks every time.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
# Load .env from the project root (experiments/llamaindex/.env).
# Must happen BEFORE os.getenv() calls below so .env values are available.
# Real OS environment variables still take precedence over .env values.
load_dotenv()

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

# --- chunking (mirrors TextChunker.java: CHUNK_SIZE=800, CHUNK_OVERLAP=100) ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# --- models (must match application.properties for a fair comparison) ---
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_DIM = 768  # nomic-embed-text output dimension

# --- postgres (same DB as the Java app, isolated table) ---
# Defaults mirror Spring's application.properties fallbacks.
PG_HOST = os.getenv("DB_HOST", "localhost")
PG_PORT = int(os.getenv("DB_PORT", "5432"))
PG_DATABASE = os.getenv("DB_NAME", "ai_rag_assistant")
PG_USER = os.getenv("DB_USERNAME", "postgres")
PG_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
PG_TABLE = "llamaindex_chunks"  # becomes `data_llamaindex_chunks` in pg


def build_embedding_model() -> OllamaEmbedding:
    return OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def build_vector_store() -> PGVectorStore:
    """Connection to the shared Postgres DB, isolated table."""
    return PGVectorStore.from_params(
        database=PG_DATABASE,
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        table_name=PG_TABLE,
        embed_dim=EMBED_DIM,
        # Enable pgvector's HNSW index for fast ANN search. LlamaIndex will
        # CREATE INDEX on first connect.
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )


def build_index(data_dir: Path) -> VectorStoreIndex:
    """
    Run the full ingestion pipeline: load -> split -> embed -> write to Postgres.
    Call once per document set (or whenever sample-data/ changes).
    """
    if not data_dir.exists() or not any(data_dir.iterdir()):
        raise FileNotFoundError(
            f"No files in {data_dir}. Drop a PDF or .txt in there first."
        )

    documents = SimpleDirectoryReader(input_dir=str(data_dir)).load_data()
    print(f"[ingest] loaded {len(documents)} document(s) from {data_dir}")

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embed_model = build_embedding_model()
    vector_store = build_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"[ingest] writing to postgres table data_{PG_TABLE} @ {PG_HOST}:{PG_PORT}/{PG_DATABASE}")
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        embed_model=embed_model,
        storage_context=storage_context,
        show_progress=True,
    )
    print("[ingest] done. chunks persisted.")
    return index


def load_index() -> VectorStoreIndex:
    """
    Reconnect to an already-populated Postgres table. No embedding work.
    This is what every --mode run calls so timing reflects retrieval only.
    """
    vector_store = build_vector_store()
    embed_model = build_embedding_model()

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    print(f"[load] connected to data_{PG_TABLE} @ {PG_HOST}:{PG_PORT}/{PG_DATABASE}")
    return index
