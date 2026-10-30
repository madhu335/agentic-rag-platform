"""
Standalone smoke test for nvidia/Nemotron-3.5-Embed-1B-BF16 vs the current BAAI/bge-base-en-v1.5.

Run inside triton-rag:26.07 (GPU, HF cache mounted, offline):
  docker run --rm --gpus all -e HF_HUB_OFFLINE=1 \
    -v C:\\Users\\Madhu\\.cache\\huggingface:/root/.cache/huggingface \
    -v C:\\dev\\ai-rag-assistant\\triton\\tools:/tools \
    --entrypoint python3 triton-rag:26.07 /tools/embed_smoke.py

Checks: loads, output dims, query/passage prompts, 1024-dim truncation + re-normalize,
ranking on a few vehicle-style queries, batch-32 latency, peak GPU memory.
"""
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

NEMOTRON = "nvidia/Nemotron-3.5-Embed-1B-BF16"
BGE = "BAAI/bge-base-en-v1.5"
DIM = 1024

DOCS = [
    "The 2025 BMW M3 Competition uses a twin-turbo 3.0L inline-six producing 503 hp and 479 lb-ft of torque.",
    "The 2025 Toyota Camry XSE Hybrid returns 47 mpg combined and has low five-year maintenance costs.",
    "The 2025 Honda CR-V Sport Hybrid offers a spacious cabin, AWD and strong resale value for families.",
    "The 2024 Porsche 911 GT3 is a track-focused coupe with a 502 hp naturally aspirated flat-six.",
    "The 2025 Tesla Model 3 Long Range has dual-motor all-wheel drive and 0-60 mph in 4.2 seconds.",
    "The 2024 Chevrolet Suburban is a full-size SUV with a 5.3L V8 and seating for up to nine.",
]
QUERIES = [
    ("How much horsepower does the BMW M3 Competition have?", 0),
    ("good family car with low running costs", 1),          # S2-002-style; 1 or 2 both reasonable
    ("best car for track days but still practical daily", 3),
    ("xyzabc vehicle", None),                                # nonsense: we want LOW top score
]


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def report(name, q, d):
    sims = q @ d.T
    print(f"\n--- {name}: dims={d.shape[1]} ---")
    for (text, expected), row in zip(QUERIES, sims):
        order = np.argsort(-row)
        top = order[0]
        mark = "" if expected is None else ("OK " if top == expected else "MISS")
        print(f"{mark:4} top={top} score={row[top]:.3f} 2nd={row[order[1]]:.3f}  | {text}")


def main():
    print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0))
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    model = SentenceTransformer(NEMOTRON, device="cuda", trust_remote_code=True,
                                model_kwargs={"torch_dtype": torch.bfloat16})
    print(f"\nloaded {NEMOTRON} in {time.time() - t0:.1f}s; prompts={model.prompts}")

    q = model.encode([t for t, _ in QUERIES], prompt_name="query", convert_to_numpy=True)
    d = model.encode(DOCS, prompt_name="document", convert_to_numpy=True)
    print("full dims:", d.shape[1], "| norms ~1:", np.round(np.linalg.norm(d, axis=1)[:3], 3))
    report("Nemotron full", q, d)
    report(f"Nemotron truncated {DIM}", l2(q[:, :DIM]), l2(d[:, :DIM]))

    # Prompt sanity: same query without the 'query: ' prefix
    q_raw = model.encode([t for t, _ in QUERIES], convert_to_numpy=True)
    report("Nemotron NO prompts (expect worse)", l2(q_raw[:, :DIM]), l2(d[:, :DIM]))

    batch = (DOCS * 6)[:32]
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3):
        model.encode(batch, prompt_name="document", batch_size=32, convert_to_numpy=True)
    torch.cuda.synchronize()
    print(f"\nbatch-32 passage encode: {(time.time() - t0) / 3 * 1000:.0f} ms avg")
    t0 = time.time()
    for _ in range(10):
        model.encode(["How much horsepower does the BMW M3 Competition have?"], prompt_name="query")
    print(f"single query encode: {(time.time() - t0) / 10 * 1000:.0f} ms avg")
    print(f"peak GPU memory (this process): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    del model
    torch.cuda.empty_cache()
    bge = SentenceTransformer(BGE, device="cuda")
    report("bge-base-en-v1.5 (current)", bge.encode([t for t, _ in QUERIES], normalize_embeddings=True),
           bge.encode(DOCS, normalize_embeddings=True))


if __name__ == "__main__":
    main()
