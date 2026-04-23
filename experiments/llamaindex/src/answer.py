"""
Answer stage.

Use the local vLLM OpenAI-compatible endpoint directly after retrieval.
This avoids LlamaIndex LLM adapter version conflicts.
"""

import httpx

VLLM_BASE_URL = "http://localhost:8001/v1"
VLLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
REQUEST_TIMEOUT = 120.0


def _build_prompt(question: str, nodes: list) -> str:
    chunks = []
    for i, node in enumerate(nodes, 1):
        text = node.node.get_content()
        chunks.append(f"{i}. {text}")

    context = "\n\n".join(chunks)

    return f"""You are a helpful assistant.
Answer only from the provided context.
If the answer is not in the context, say: "I don't know based on the retrieved context."

Context:
{context}

Question:
{question}

Answer:"""


def _call_vllm(prompt: str) -> str:
    response = httpx.post(
        f"{VLLM_BASE_URL}/chat/completions",
        json={
            "model": VLLM_MODEL,
            "temperature": 0.2,
            "max_tokens": 512,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()


def answer_with_vllm(question: str, nodes: list) -> str:
    prompt = _build_prompt(question, nodes)
    return _call_vllm(prompt)