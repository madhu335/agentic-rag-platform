"""
Re-embed all stored passages with Triton's nemotron_embed (INPUT_TYPE=passage).

Reads rows whose embedding IS NULL, embeds BATCH texts per Triton call, writes the vectors back.
The halfvec triggers fill embedding_half. Restartable: finished rows are no longer NULL.

Run from the triton-rag image (needs psycopg; installed on the fly):
  docker run --rm -v C:\\dev\\ai-rag-assistant\\scripts\\nemotron-embed:/work --entrypoint bash triton-rag:26.07 \
    -c "pip install -q 'psycopg[binary]' && python3 /work/reembed.py"
Env overrides: PG_DSN, TRITON_URL, MODEL, BATCH.
"""
import json
import os
import sys
import time
import urllib.request

import psycopg

PG_DSN = os.environ.get("PG_DSN", "host=host.docker.internal port=5432 dbname=ai_rag_assistant user=postgres password=postgres")
TRITON_URL = os.environ.get("TRITON_URL", "http://host.docker.internal:8000")
MODEL = os.environ.get("MODEL", "nemotron_embed")
BATCH = int(os.environ.get("BATCH", "64"))
EXPECTED_DIM = int(os.environ.get("EXPECTED_DIM", "1024"))

TABLES = [
    # (label, select sql, update sql)
    ("document_chunks",
     "SELECT id, doc_type, coalesce(content, '') FROM document_chunks WHERE embedding IS NULL ORDER BY doc_type, id LIMIT %s",
     "UPDATE document_chunks SET embedding = %s::vector WHERE id = %s AND doc_type = %s"),
    ("vehicle_summaries",
     "SELECT vehicle_id, NULL, coalesce(summary_text, '') FROM vehicle_summaries WHERE embedding IS NULL ORDER BY vehicle_id LIMIT %s",
     "UPDATE vehicle_summaries SET embedding = %s::vector WHERE vehicle_id = %s"),
]


def embed_passages(texts):
    body = {
        "inputs": [
            {"name": "TEXT", "shape": [len(texts), 1], "datatype": "BYTES", "data": [[t] for t in texts]},
            {"name": "INPUT_TYPE", "shape": [len(texts), 1], "datatype": "BYTES", "data": [["passage"]] * len(texts)},
        ]
    }
    req = urllib.request.Request(
        f"{TRITON_URL}/v2/models/{MODEL}/infer",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read())["outputs"][0]
    n, dim = out["shape"]
    if dim != EXPECTED_DIM:
        raise RuntimeError(f"Triton returned dim={dim}, expected {EXPECTED_DIM}")
    flat = out["data"]
    return [flat[i * dim:(i + 1) * dim] for i in range(n)]


def to_pg(vec):
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


def main():
    with psycopg.connect(PG_DSN) as conn:
        for label, select_sql, update_sql in TABLES:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {label} WHERE embedding IS NULL")
                total = cur.fetchone()[0]
            print(f"{label}: {total} rows to embed", flush=True)
            done, t0 = 0, time.time()
            while True:
                with conn.cursor() as cur:
                    cur.execute(select_sql, (BATCH,))
                    rows = cur.fetchall()
                if not rows:
                    break
                vectors = embed_passages([r[2] for r in rows])
                with conn.cursor() as cur:
                    if label == "document_chunks":
                        cur.executemany(update_sql, [(to_pg(v), r[0], r[1]) for v, r in zip(vectors, rows)])
                    else:
                        cur.executemany(update_sql, [(to_pg(v), r[0]) for v, r in zip(vectors, rows)])
                conn.commit()
                done += len(rows)
                if done % (BATCH * 20) < BATCH or done == total:
                    rate = done / max(time.time() - t0, 1e-6)
                    print(f"  {label}: {done}/{total}  ({rate:.0f} rows/s)", flush=True)
            print(f"{label}: done in {time.time() - t0:.0f}s", flush=True)

        with conn.cursor() as cur:
            cur.execute("SELECT doc_type, count(*), count(embedding), count(embedding_half) FROM document_chunks GROUP BY 1 ORDER BY 1")
            for row in cur.fetchall():
                print("check document_chunks", row)
            cur.execute("SELECT count(*), count(embedding), count(embedding_half), max(vector_dims(embedding)) FROM vehicle_summaries")
            print("check vehicle_summaries (rows, embedded, half, dims)", cur.fetchone())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FAILED:", e, file=sys.stderr)
        sys.exit(1)
