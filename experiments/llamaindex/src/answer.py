"""
Answer stage.

Java equivalent: RagAnswerService — takes retrieved chunks, formats a prompt,
calls Ollama llama3.1, returns grounded answer.

LlamaIndex equivalent (this file): RetrieverQueryEngine.from_args(retriever, llm).
One line. That's the whole point of the framework.
"""

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.ollama import Ollama

# Same LLM as application.properties `ollama.model=llama3.1`
LLM_MODEL = "llama3.1"
OLLAMA_BASE_URL = "http://localhost:11434"
REQUEST_TIMEOUT = 120.0


def build_llm() -> Ollama:
    return Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=REQUEST_TIMEOUT,
    )


def build_query_engine(retriever: BaseRetriever) -> RetrieverQueryEngine:
    llm = build_llm()
    # TODO experiment: add node_postprocessors=[SentenceTransformerRerank(...)]
    # to get a rough analogue of your Java HYBRID_RERANK mode.
    return RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)
