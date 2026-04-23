"""
Ingestion stage (Postgres + pgvector + Triton embeddings).

Robust version:
- Handles Triton instability
- Limits batch size
- Truncates large text
- Retries on failure
"""

import os
import time
from pathlib import Path
from typing import List

import httpx
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore


# ---------------- CONFIG ----------------

CHUNK_SIZE = 256            # 🔥 reduced for stability
CHUNK_OVERLAP = 50

TRITON_BASE_URL = os.getenv("TRITON_BASE_URL", "http://localhost:8000")
TRITON_EMBED_MODEL = os.getenv("TRITON_EMBED_MODEL", "text_embedding")
TRITON_TIMEOUT = float(os.getenv("TRITON_TIMEOUT", "120"))

EMBED_DIM = 768

MAX_BATCH_SIZE = 4          # 🔥 critical fix
MAX_CHARS = 2000           # 🔥 truncate long text
RETRIES = 2                # 🔥 retry on failure


# ---------------- POSTGRES ----------------

PG_HOST = os.getenv("DB_HOST", "localhost")
PG_PORT = int(os.getenv("DB_PORT", "5432"))
PG_DATABASE = os.getenv("DB_NAME", "ai_rag_assistant")
PG_USER = os.getenv("DB_USERNAME", "postgres")
PG_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

PG_TABLE = "llamaindex_chunks"


# ---------------- EMBEDDING ----------------

class TritonEmbedding(BaseEmbedding):
    model_name: str = TRITON_EMBED_MODEL
    base_url: str = TRITON_BASE_URL
    timeout: float = TRITON_TIMEOUT

    def _infer_batch(self, texts: List[str]) -> List[List[float]]:
        response = httpx.post(
            f"{self.base_url}/v2/models/{self.model_name}/infer",
            json={
                "inputs": [
                    {
                        "name": "TEXT",
                        "shape": [len(texts), 1],
                        "datatype": "BYTES",
                        "data": [[t] for t in texts],
                    }
                ]
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        outputs = payload.get("outputs", [])
        if not outputs:
            raise RuntimeError("Missing Triton outputs")

        embedding_output = next(
            (o for o in outputs if o.get("name") == "EMBEDDING"),
            outputs[0],
        )

        data = embedding_output.get("data", [])
        shape = embedding_output.get("shape", [])

        if not data:
            raise RuntimeError("Missing embedding data")

        if len(shape) != 2 or shape[1] != EMBED_DIM:
            raise RuntimeError(f"Unexpected embedding shape: {shape}")

        batch_size = shape[0]
        return [
            data[i * EMBED_DIM : (i + 1) * EMBED_DIM]
            for i in range(batch_size)
        ]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # 🔥 sanitize + truncate
        safe_texts = [
            (t or "").replace("\x00", " ").strip()[:MAX_CHARS]
            for t in texts
        ]

        all_embeddings: List[List[float]] = []

        for i in range(0, len(safe_texts), MAX_BATCH_SIZE):
            batch = safe_texts[i:i + MAX_BATCH_SIZE]

            for attempt in range(RETRIES + 1):
                try:
                    result = self._infer_batch(batch)
                    all_embeddings.extend(result)
                    break
                except Exception as e:
                    if attempt < RETRIES:
                        print(f"[embed] retry {attempt+1}/{RETRIES} for batch size={len(batch)}")
                        time.sleep(1)
                    else:
                        print(f"[embed] batch failed → fallback to single: {e}")
                        for text in batch:
                            try:
                                single = self._infer_batch([text])[0]
                                all_embeddings.append(single)
                            except Exception as single_e:
                                preview = text[:120].replace("\n", " ")
                                raise RuntimeError(
                                    f"Triton failed on text: '{preview}'"
                                ) from single_e

        return all_embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_batch([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)


# ---------------- BUILDERS ----------------

def build_embedding_model() -> TritonEmbedding:
    return TritonEmbedding()


def build_vector_store() -> PGVectorStore:
    return PGVectorStore.from_params(
        database=PG_DATABASE,
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        table_name=PG_TABLE,
        embed_dim=EMBED_DIM,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )


# ---------------- PIPELINE ----------------

def build_index(data_dir: Path) -> VectorStoreIndex:
    if not data_dir.exists() or not any(data_dir.iterdir()):
        raise FileNotFoundError(f"No files in {data_dir}")

    documents = SimpleDirectoryReader(input_dir=str(data_dir)).load_data()
    print(f"[ingest] loaded {len(documents)} document(s)")

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    embed_model = build_embedding_model()
    vector_store = build_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"[ingest] writing to data_{PG_TABLE}")

    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        embed_model=embed_model,
        storage_context=storage_context,
        show_progress=True,
    )

    print("[ingest] done")
    return index


def load_index() -> VectorStoreIndex:
    vector_store = build_vector_store()
    embed_model = build_embedding_model()

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    print(f"[load] connected to data_{PG_TABLE}")
    return index