"""
Judge eval on REAL questions: how often does the cascade escalate, and do the two judges agree?

The app's existing evals don't call the judge (recall reports = retrieval only; /api/eval/answer = phrase match,
bypasses the orchestrator). This script does:

  1. generate  For each vehicle golden-set question: retrieve context through the app (POST /vehicles/ask,
               fleet search, top-K chunk excerpts), generate an answer with Nemotron on Triton using the same
               prompt as TritonVllmClient (thinking off, temperature 0.0). Saves cases to a JSON file so every
               judge grades IDENTICAL answers.
  2. judge     Send each saved case to POST /api/judge/test (whatever judge.provider the app runs with).
               Saves results and prints verdicts, cascade stages, escalation rate and latency.
  3. compare   Two results files (e.g. halo vs triton): accept/reject agreement and every disagreement.

Note: /vehicles/ask returns 200-character excerpts, so context is shorter than the app's real RAG context.
Answers are generated from that same context, so the judge sees exactly what the answer was based on.

Standard library only. Python 3.9+. Run from C:\\dev\\ai-rag-assistant (app running on :8080, triton-rag up).

  python scripts\\judge-compare\\judge_eval.py generate
  python scripts\\judge-compare\\judge_eval.py judge --label triton      (app started with JUDGE_PROVIDER=triton)
  python scripts\\judge-compare\\judge_eval.py judge --label triton-thorough --mode thorough   (sync-path judge)
  python scripts\\judge-compare\\judge_eval.py judge --label halo        (app restarted with JUDGE_PROVIDER=halo)
  python scripts\\judge-compare\\judge_eval.py compare --a halo --b triton
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

APP = "http://localhost:8080"
TRITON = "http://localhost:8000/v2/models/nemotron/generate"
GOLDEN = os.path.join("src", "main", "resources", "eval", "vehicle_golden_set.json")
OUT_DIR = os.path.join("scripts", "judge-compare", "eval-out")
CASES = os.path.join(OUT_DIR, "cases.json")


def post(url, body, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def nemotron_answer(question, chunks):
    """Same prompt as TritonVllmClient: system + numbered context + question, thinking off."""
    user = []
    if chunks:
        user.append("Context:")
        user += [f"{i}. {c}" for i, c in enumerate(chunks, 1)]
        user.append("")
    user += ["Question:", question, "", "Answer:"]
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n" + "\n".join(user) + "<|im_end|>\n<|im_start|>assistant\n<think></think>")
    out = post(TRITON, {"text_input": prompt, "parameters": {
        "stream": False, "exclude_input_in_output": True, "max_tokens": 512, "temperature": 0.0}})
    text = out.get("text_output", "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<|im_end|>", "").strip()


def cmd_generate(args):
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    if args.limit:
        golden = golden[:args.limit]
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = []
    for i, g in enumerate(golden, 1):
        q = g["query"]
        try:
            hits = post(f"{APP}/vehicles/ask", {"question": q, "topK": args.top_k})
            chunks = [h["excerpt"] for h in hits]
            answer = nemotron_answer(q, chunks) if chunks else "I don't know."
            cases.append({"id": g["entryId"], "category": g.get("category"), "question": q,
                          "context": chunks, "answer": answer})
            print(f"[{i:>2}/{len(golden)}] {g['entryId']:<10} {len(chunks)} chunks | {answer[:90]!r}")
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            print(f"[{i:>2}/{len(golden)}] {g['entryId']:<10} SKIPPED: {e}")
    with open(CASES, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    print(f"\nSaved {len(cases)} cases -> {CASES}")


def pct(n, d):
    return f"{n}/{d} ({100 * n / max(d, 1):.0f}%)"


def cmd_judge(args):
    with open(CASES, encoding="utf-8") as f:
        cases = json.load(f)
    results = []
    for i, c in enumerate(cases, 1):
        t0 = time.perf_counter()
        try:
            r = post(f"{APP}/api/judge/test",
                     {"question": c["question"], "answer": c["answer"], "context": c["context"], "mode": args.mode})
            res = r["result"]
            row = {"id": c["id"], "category": c["category"], "provider": r.get("provider"),
                   "latencyMs": r.get("latencyMs"), "verdict": res.get("verdict"), "score": res.get("score"),
                   "stage": res.get("stage"), "reason": res.get("reason"),
                   "questionRisk": r.get("questionRisk"), "answerChecks": r.get("answerChecks")}
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            row = {"id": c["id"], "category": c["category"], "error": str(e),
                   "latencyMs": int((time.perf_counter() - t0) * 1000)}
        results.append(row)
        print(f"[{i:>2}/{len(cases)}] {row['id']:<10} {row.get('verdict', 'ERROR'):<12} "
              f"{str(row.get('stage', row.get('error')))[:48]:<48} {row['latencyMs']:>6} ms | {row.get('reason', '')}")

    path = os.path.join(OUT_DIR, f"results-{args.label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    ok = [r for r in results if "error" not in r]
    lat = [r["latencyMs"] for r in ok]
    stages = Counter(("escalated" if str(r["stage"]).startswith("escalated") else r["stage"]) for r in ok)
    esc_why = Counter(str(r["stage"]).split(":")[1] for r in ok if str(r["stage"]).startswith("escalated"))
    print(f"\n== {args.label}  (provider={ok[0]['provider'] if ok else '?'}, mode={args.mode})  saved -> {path}")
    print(f"   verdicts : {dict(Counter(r['verdict'] for r in ok))}")
    print(f"   accepted : {pct(sum(r['verdict'] == 'PASS' for r in ok), len(ok))}   errors: {len(results) - len(ok)}")
    print(f"   stages   : {dict(stages)}")
    if esc_why:
        print(f"   escalated: {pct(sum(esc_why.values()), len(ok))}  why: {dict(esc_why)}")
    if ok and ok[0].get("questionRisk") is not None:
        sync = [r for r in ok if r["questionRisk"] != "none" or r["answerChecks"] != "none"]
        print(f"   orchestrator would judge synchronously (risky question or check fired): {pct(len(sync), len(ok))}")
    if lat:
        lat_sorted = sorted(lat)
        p95 = lat_sorted[min(len(lat_sorted) - 1, int(round(0.95 * (len(lat_sorted) - 1))))]
        print(f"   latency  : avg {statistics.mean(lat):.0f} ms | p50 {statistics.median(lat):.0f} ms | p95 {p95} ms")
        by_stage = {}
        for r in ok:
            key = "escalated" if str(r["stage"]).startswith("escalated") else r["stage"]
            by_stage.setdefault(key, []).append(r["latencyMs"])
        print("   by stage : " + " | ".join(f"{k} avg {statistics.mean(v):.0f} ms" for k, v in by_stage.items()))


def cmd_compare(args):
    def load(label):
        with open(os.path.join(OUT_DIR, f"results-{label}.json"), encoding="utf-8") as f:
            return {r["id"]: r for r in json.load(f) if "error" not in r}
    a, b = load(args.a), load(args.b)
    ids = [i for i in a if i in b]
    with open(CASES, encoding="utf-8") as f:
        cases = {c["id"]: c for c in json.load(f)}
    same = sum((a[i]["verdict"] == "PASS") == (b[i]["verdict"] == "PASS") for i in ids)
    print(f"== {args.a} vs {args.b}: accept/reject agreement {pct(same, len(ids))}, "
          f"exact verdict agreement {pct(sum(a[i]['verdict'] == b[i]['verdict'] for i in ids), len(ids))}\n")
    m = Counter((a[i]["verdict"] == "PASS", b[i]["verdict"] == "PASS") for i in ids)
    print(f"   {'':<22}{args.b + ' ACCEPT':>16}{args.b + ' REJECT':>16}")
    print(f"   {args.a + ' ACCEPT':<22}{m[(True, True)]:>16}{m[(True, False)]:>16}")
    print(f"   {args.a + ' REJECT':<22}{m[(False, True)]:>16}{m[(False, False)]:>16}\n")
    for i in ids:
        if (a[i]["verdict"] == "PASS") != (b[i]["verdict"] == "PASS"):
            c = cases.get(i, {})
            print(f"-- {i} [{a[i]['category']}] {c.get('question')}")
            print(f"   answer : {c.get('answer', '')[:200]}")
            print(f"   {args.a:<7}: {a[i]['verdict']:<12} {a[i]['reason']}")
            print(f"   {args.b:<7}: {b[i]['verdict']:<12} {b[i]['reason']}  ({b[i]['stage']})\n")


def main():
    p = argparse.ArgumentParser(description="Judge eval on real golden-set questions")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--top-k", type=int, default=5)
    g.add_argument("--limit", type=int, default=0, help="only the first N questions")
    j = sub.add_parser("judge")
    j.add_argument("--label", required=True, help="name for this run, e.g. triton or halo")
    j.add_argument("--mode", default="auto", choices=["auto", "thorough"],
                   help="auto = cascade/async-audit path, thorough = the orchestrator's sync path")
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    args = p.parse_args()
    if not os.path.exists(GOLDEN):
        sys.exit(f"Run from C:\\dev\\ai-rag-assistant ({GOLDEN} not found)")
    {"generate": cmd_generate, "judge": cmd_judge, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
