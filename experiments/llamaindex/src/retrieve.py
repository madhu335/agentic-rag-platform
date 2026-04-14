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


def _load_all_nodes(index: VectorStoreIndex) -> list:
    """
    Pull every node out of the pgvector-backed store so BM25Retriever has
    something to build its in-memory index over. For small-to-medium corpora
    this is fine; for very large ones you'd want a different BM25 strategy
    (e.g. Postgres FTS, which is what your Java side already does).
    """
    vector_store = index.vector_store

    # Trick: query with a zero vector and a huge top_k to grab everything.
    # LlamaIndex's PGVectorStore doesn't expose a "get all nodes" API, so this
    # is the pragmatic workaround. The dummy vector doesn't matter because we
    # only care about the node content, not the similarity ordering.
    dummy_query = VectorStoreQuery(
        query_embedding=[0.0] * 768,  # nomic-embed-text dim
        similarity_top_k=10_000,
    )
    result = vector_store.query(dummy_query)
    return result.nodes or []


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
