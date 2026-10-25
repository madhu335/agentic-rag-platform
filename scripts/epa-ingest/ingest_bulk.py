#!/usr/bin/env python3
"""
ingest_bulk.py — stream vehicles into POST /vehicles/ingest/bulk in batches,
with resumable checkpoints and a throughput log you can use as a benchmark.

Standard library only.

Usage (from C:\\dev\\ai-rag-assistant, app running on 8080):
    python scripts\\epa-ingest\\ingest_bulk.py --limit 200           # smoke test first
    python scripts\\epa-ingest\\ingest_bulk.py                       # everything, resumes automatically
    python scripts\\epa-ingest\\ingest_bulk.py --fresh               # ignore checkpoint, start over

Input : data/epa/vehicles_enriched.jsonl if it exists, else data/epa/vehicles.jsonl
Output: data/epa/ingest_checkpoint.json   (resume point)
        data/epa/ingest_metrics.csv       (one row per batch: latency, vehicles/s, chunks/s)

Why batches of 50: BulkVehicleIngestionService pages at 50, so each HTTP call is
exactly one server page (chunk -> batch embed -> batch upsert -> summaries).
One request per page also makes the checkpoint granular: a crash at vehicle
9,000 resumes at vehicle 9,000, not 0. Re-sending a batch is safe because
PgVectorStore upserts ON CONFLICT (doc_id, chunk_index, doc_type) and summaries
upsert by vehicle_id — at-least-once delivery with idempotent writes.
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_vehicles(path: Path, limit: int):
    vehicles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                vehicles.append(json.loads(line))
                if limit and len(vehicles) >= limit:
                    break
    return vehicles


def post_batch(url: str, batch, timeout: int):
    body = json.dumps(batch).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--in", dest="inp", default="")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="only the first N vehicles (0 = all)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per batch request")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--fresh", action="store_true", help="ignore the checkpoint and start from vehicle 0")
    ap.add_argument("--data-dir", default="data/epa")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    inp = Path(args.inp) if args.inp else (
        data_dir / "vehicles_enriched.jsonl" if (data_dir / "vehicles_enriched.jsonl").exists()
        else data_dir / "vehicles.jsonl")
    ckpt_path = data_dir / "ingest_checkpoint.json"
    metrics_path = data_dir / "ingest_metrics.csv"
    url = args.base_url.rstrip("/") + "/vehicles/ingest/bulk"

    vehicles = load_vehicles(inp, args.limit)
    total = len(vehicles)
    if total == 0:
        sys.exit(f"No vehicles in {inp}")

    start = 0
    if ckpt_path.exists() and not args.fresh:
        ckpt = json.loads(ckpt_path.read_text())
        if ckpt.get("input") == str(inp):
            start = ckpt.get("next_index", 0)
    if start >= total:
        print(f"Checkpoint says all {total} vehicles are already ingested. Use --fresh to redo.")
        return

    print(f"Ingesting {total - start} of {total} vehicles from {inp}")
    print(f"  -> {url}  (batch {args.batch_size}, starting at index {start})")

    new_metrics = not metrics_path.exists() or args.fresh
    mf = open(metrics_path, "w" if new_metrics else "a", newline="")
    mw = csv.writer(mf)
    if new_metrics:
        mw.writerow(["batch_start", "vehicles", "chunks", "summaries", "server_ms",
                     "wall_ms", "vehicles_per_s", "chunks_per_s", "errors"])

    run_t0 = time.time()
    done_v = done_c = done_s = 0
    batch_ms, all_errors = [], []

    try:
        for i in range(start, total, args.batch_size):
            batch = vehicles[i:i + args.batch_size]
            result, last_err = None, None
            for attempt in range(1, args.retries + 1):
                t0 = time.time()
                try:
                    result = post_batch(url, batch, args.timeout)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                    detail = ""
                    if isinstance(e, urllib.error.HTTPError):
                        detail = e.read().decode("utf-8", "replace")[:300]
                    last_err = f"{e} {detail}".strip()
                    wait = 5 * attempt
                    print(f"  batch @{i} attempt {attempt} failed: {last_err} — retrying in {wait}s")
                    time.sleep(wait)
            if result is None:
                print(f"\nStopping: batch @{i} failed {args.retries} times. Last error: {last_err}")
                print("Fix the cause and re-run; it resumes from this batch.")
                break

            wall_ms = (time.time() - t0) * 1000
            v = result.get("totalVehicles", 0)
            c = result.get("totalChunks", 0)
            s = result.get("totalSummaries", 0)
            server_ms = result.get("elapsedMs", 0)
            errs = result.get("errors") or []
            secs = max(wall_ms / 1000, 1e-6)
            mw.writerow([i, v, c, s, server_ms, round(wall_ms), round(v / secs, 2), round(c / secs, 1), len(errs)])
            mf.flush()

            done_v += v; done_c += c; done_s += s
            batch_ms.append(wall_ms)
            all_errors.extend(errs)

            # The server returns HTTP 200 even when a page fails internally (errors in
            # the body). Never checkpoint past a batch that wrote nothing, or a re-run
            # would skip vehicles that were never ingested.
            if c == 0 and errs:
                print(f"\nStopping: batch @{i} wrote 0 chunks. Server errors:")
                for e in errs[:5]:
                    print(f"    - {e}")
                print("Checkpoint NOT advanced. Fix the cause and re-run; it resumes from this batch.")
                break
            if s < v:
                print(f"  WARNING: batch @{i}: {v - s} vehicles got chunks but no Tier-1 summary")

            ckpt_path.write_text(json.dumps({"input": str(inp), "next_index": i + len(batch),
                                             "updated": time.strftime("%Y-%m-%d %H:%M:%S")}))

            elapsed = time.time() - run_t0
            rate = done_v / elapsed if elapsed else 0
            remaining = total - (i + len(batch))
            eta = remaining / rate if rate else 0
            print(f"  {i + len(batch):>6}/{total}  +{v} vehicles, +{c} chunks, +{s} summaries  "
                  f"{wall_ms:7.0f} ms  | {rate:5.1f} veh/s  ETA {eta / 60:5.1f} min"
                  + (f"  ({len(errs)} errors)" if errs else ""))
    except KeyboardInterrupt:
        print("\nInterrupted — progress is checkpointed; re-run to resume.")
    finally:
        mf.close()

    elapsed = time.time() - run_t0
    if batch_ms:
        p50 = statistics.median(batch_ms)
        p95 = sorted(batch_ms)[min(len(batch_ms) - 1, math.ceil(len(batch_ms) * 0.95) - 1)]
        print("\n── Ingestion summary ─────────────────────────────")
        print(f"  vehicles: {done_v}   chunks: {done_c}   summaries: {done_s}")
        print(f"  wall time: {elapsed / 60:.1f} min   throughput: {done_v / elapsed:.1f} veh/s, "
              f"{done_c / elapsed:.0f} chunks/s")
        print(f"  batch latency p50: {p50:.0f} ms   p95: {p95:.0f} ms")
        print(f"  server-reported errors: {len(all_errors)}")
        for e in all_errors[:10]:
            print(f"    - {e}")
        if done_s < done_v:
            print(f"  WARNING: {done_v - done_s} vehicles have no summary -> invisible to Tier 1 fleet search.")
        print(f"  metrics: {metrics_path}")
        print("\nNext: run post_load.sql to rebuild the IVFFlat index, then re-run the eval.")


if __name__ == "__main__":
    main()
