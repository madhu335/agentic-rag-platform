"""
Retrieval stage.

Java equivalent: RagRetriever with RetrievalMode.{VECTOR, BM25, HYBRID, HYBRID_RERANK}
backed by PgVectorStore.hybridSearch() which does SQL-level fusion of pgvector
cosine distance and Postgres ts_rank_cd BM25.

LlamaIndex equivalent (this file):
    VectorIndexRetriever  -> VECTOR mode (hits pgvector directly)
    BM25Retriever         -> BM25 mode (in-Python, over nodes pulled from pg)
    QueryFusionRetriever  -> HYBRID mode (reciprocal rank fusion)

Important subtlety now that the store is Postgres, not in-memory:

- VectorIndexRetriever queries pg directly every call. Fast, scales with pgvector's
  HNSW index. This is the apples-to-apples race against your Java VECTOR mode.
- BM25Retriever does NOT live in pg. It builds an in-memory rank_bm25 index from
  nodes read out of pg once at startup. Your Java BM25 uses Postgres ts_vector /
  ts_rank_cd — a fundamentally different implementation. Put this in notes.md;
  it's one of the more interesting findings of the whole experiment.
"""

from typing import Literal

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    VectorIndexRetriever,
)
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.retrievers.bm25 import BM25Retriever

Mode = Literal["vector", "bm25", "hybrid"]

DEFAULT_TOP_K = 5


_ALL_NODES_CACHE = None

def _load_all_nodes(index: VectorStoreIndex) -> list:
    global _ALL_NODES_CACHE
    if _ALL_NODES_CACHE is not None:
        return _ALL_NODES_CACHE

    vector_store = index.vector_store

    dummy_query = VectorStoreQuery(
        query_embedding=[0.0] * 768,
        similarity_top_k=10_000,
    )
    result = vector_store.query(dummy_query)

    _ALL_NODES_CACHE = result.nodes or []
    return _ALL_NODES_CACHE


def build_retriever(index: VectorStoreIndex, mode: Mode, top_k: int = DEFAULT_TOP_K) -> BaseRetriever:
    """Factory that mirrors RagRetriever.retrieve(docId, question, topK, mode)."""

    if mode == "vector":
        # Straight cosine-similarity retrieval against pgvector.
        # This IS the fair comparison for your Java VECTOR mode.
        return VectorIndexRetriever(index=index, similarity_top_k=top_k)

    if mode == "bm25":
        nodes = _load_all_nodes(index)
        if not nodes:
            raise RuntimeError(
                "No nodes in pg. Did you run `python -m src.run_pipeline --ingest` first?"
            )
        return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

    if mode == "hybrid":
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        nodes = _load_all_nodes(index)
        if not nodes:
            raise RuntimeError(
                "No nodes in pg. Did you run `python -m src.run_pipeline --ingest` first?"
            )
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

        # Reciprocal rank fusion. num_queries=1 disables query rewriting so the
        # comparison stays honest against your Java hybridSearch.
        return QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )

    raise ValueError(f"unknown retrieval mode: {mode}")
