"""
Head-to-head judge comparison: Nemotron-3-Nano-30B (Triton on the 5090) vs gpt-oss-120b (llama.cpp on GN100).

Tests the judge CONTRACT before changing the Java code:

  --prompt current   today's JudgeService prompt (grounded/correct/complete booleans + score, example values
                     true/true/true/0.7). Decision = DefaultResultEvaluationPolicy logic:
                     accept = grounded && correct && score >= 0.70 && !(complete == false && score < 0.75)
  --prompt verdict   proposed prompt: ONE verdict field (PASS | UNSUPPORTED | INCORRECT | INCOMPLETE) with
                     definitions, placeholders instead of example values. Decision = verdict == PASS.

Per model it reports:
  decision accuracy   accept/reject matches what the app SHOULD do (the metric that matters)
  category accuracy   (verdict prompt) the failure type is one of the acceptable ones
  conflicts           verdict vs score disagree (the proposed consistency guard would re-judge these)
  valid JSON, latency

It also runs the deterministic NUMBERS CHECK once per case (every number in the answer must appear in the
context) to show how well it works as an escalation trigger for a reasoning judge.

Standard library only. Python 3.9+.

Usage (from C:\\dev\\ai-rag-assistant):
  python scripts\\judge-compare\\judge_compare.py --prompt verdict
  python scripts\\judge-compare\\judge_compare.py --prompt verdict --nemotron-think --max-tokens 3072 --runs 3
  python scripts\\judge-compare\\judge_compare.py --prompt current                 # baseline (today's app)
  python scripts\\judge-compare\\judge_compare.py --only swapped_numbers -v        # one case, raw outputs
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.request

NEMOTRON_URL = "http://localhost:8000/v2/models/nemotron/generate"
GPTOSS_URL = "http://192.168.50.20:18002/v1/chat/completions"
GPTOSS_MODEL = "gpt-oss-120b-Q4_K_M"

M3 = ("The 2025 BMW M3 Competition uses a twin-turbo 3.0L inline-six producing 503 hp and 479 lb-ft of torque. "
      "It is offered with rear-wheel drive or xDrive all-wheel drive.")
CAMRY = "The 2025 Toyota Camry XSE Hybrid returns 47 mpg combined and has low five-year maintenance costs."
PRICE = "The 2025 BMW M3 Competition xDrive has an MSRP of $81,900. The destination charge is $1,175."

PASS_ONLY = {"PASS"}
# name, skill, question, answer, context, should_accept, acceptable verdicts (verdict prompt)
CASES = [
    ("supported", "accept a correct, supported answer",
     "How much horsepower does the BMW M3 Competition have?",
     "The BMW M3 Competition produces 503 hp.", [M3], True, PASS_ONLY),
    ("paraphrase", "not over-strict on wording",
     "How much horsepower does the BMW M3 Competition have?",
     "A little over five hundred horsepower, from a twin-turbo straight-six.", [M3], True, PASS_ONLY),
    ("unsupported_claim", "catch one invented fact among true ones",
     "How much horsepower does the BMW M3 Competition have?",
     "It makes 503 hp and does 0-60 mph in 3.0 seconds.", [M3], False, {"UNSUPPORTED"}),
    ("swapped_numbers", "catch numbers attached to the wrong quantity",
     "What are the horsepower and torque of the BMW M3 Competition?",
     "It produces 479 hp and 503 lb-ft of torque.", [M3], False, {"UNSUPPORTED", "INCORRECT"}),
    ("arithmetic_right", "verify a derived number (sum)",
     "What is the total price of the M3 Competition xDrive including destination?",
     "$83,075 ($81,900 MSRP plus $1,175 destination).", [PRICE], True, PASS_ONLY),
    ("arithmetic_wrong", "catch a wrong derived number",
     "What is the total price of the M3 Competition xDrive including destination?",
     "$84,075 ($81,900 MSRP plus $1,175 destination).", [PRICE], False, {"INCORRECT", "UNSUPPORTED"}),
    ("wrong_context", "notice the context does not cover the question",
     "How much horsepower does the BMW M3 Competition have?",
     "The BMW M3 Competition produces 503 hp.", [CAMRY], False, {"UNSUPPORTED"}),
    ("honest_idk", "accept an honest 'not in context' answer",
     "How much horsepower does the BMW M3 Competition have?",
     "The provided context does not say how much horsepower the BMW M3 Competition has.", [CAMRY],
     True, PASS_ONLY),
]

SYSTEM = ("Reasoning: {effort}\nYou are a strict evaluator for a retrieval-augmented QA system. "
          "Judge the answer only against the provided context. Respond with a single JSON object and nothing else.")

# Today's JudgeService.buildPrompt, verbatim.
PROMPT_CURRENT = """You are an evaluator.

Evaluate the answer using only the provided context.

Question:
{question}

Answer:
{answer}

Decide quickly: check each claim in the answer against the context once,
then give your verdict. Do not deliberate at length.

Return JSON only in this exact format:
{{
  "grounded": true,
  "correct": true,
  "complete": true,
  "score": 0.7,
  "reason": "max 12 words, name the unsupported claim if any"
}}

Rules:
- score must be a decimal between 0.0 and 1.0
- reason must be at most 12 words
- do NOT use a 1-10 scale
- do NOT use percentages
- do NOT return markdown
- do NOT return any text outside the JSON
"""

# Proposed prompt: one verdict, defined categories, placeholders (no example values to copy).
PROMPT_VERDICT = """You are an evaluator.

Evaluate the answer using only the provided context.

Question:
{question}

Answer:
{answer}

Choose exactly one verdict:
- UNSUPPORTED: the answer states a fact, number or detail the context does not state,
  or attaches a number from the context to the wrong quantity.
- INCORRECT: the facts used are in the context, but the answer derives something wrongly
  (a wrong total, comparison or conclusion).
- INCOMPLETE: everything stated is supported and correct, but the answer leaves out part of
  what the question asked although the context contains it.
- PASS: every claim is supported and correct, and the question is answered. An answer that
  honestly says the context does not contain the information is PASS when that is true.

Check each claim in the answer against the context once, then decide.

Return JSON only, in this format:
{{"verdict": "<PASS|UNSUPPORTED|INCORRECT|INCOMPLETE>", "score": <0.0 to 1.0, where 1.0 is fully correct and supported>, "reason": "<at most 12 words; name the problem claim if any>"}}

Rules:
- do NOT return markdown
- do NOT return any text outside the JSON
"""

VERDICTS = {"PASS", "UNSUPPORTED", "INCORRECT", "INCOMPLETE"}


def user_message(template, question, answer, context):
    ctx = "\n\n---\n\n".join(context) if context else "(no context provided)"
    return "CONTEXT:\n" + ctx + "\n\n" + template.format(question=question, answer=answer)


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def call_nemotron(system, user, args):
    prompt = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
              f"<|im_start|>assistant\n" + ("<think>\n" if args.nemotron_think else "<think></think>"))
    body = {"text_input": prompt, "parameters": {"stream": False, "exclude_input_in_output": True,
                                                  "max_tokens": args.max_tokens, "temperature": args.temperature}}
    data = post(NEMOTRON_URL, body, args.timeout)
    text = data.get("text_output", "")
    # Triton's generate response has no finish_reason. With thinking on, a missing </think> means the
    # reasoning was still running when max_tokens ran out, so no JSON was ever written.
    if args.nemotron_think and "</think>" not in text:
        return text.strip(), "length (reasoning cut off: no </think>, raise --max-tokens)"
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<|im_end|>", "").strip(), "stop"


def call_gptoss(system, user, args):
    body = {"model": GPTOSS_MODEL, "temperature": args.temperature, "max_tokens": args.max_tokens, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    data = post(GPTOSS_URL, body, args.timeout)
    choice = data["choices"][0]
    text = choice["message"].get("content") or ""
    if "<|channel|>final<|message|>" in text:          # same cleanup as HaloJudgeClient.stripReasoning
        text = text.rsplit("<|channel|>final<|message|>", 1)[1]
    text = re.sub(r"(?s)<think>.*?</think>", "", text).replace("<|end|>", "").replace("<|return|>", "")
    return text.strip(), choice.get("finish_reason")


def extract_json(text):
    """Same rule as JudgeService.extractJsonObject: strip ``` fences, take first '{' .. last '}'."""
    t = re.sub(r"^```(json)?\s*|\s*```$", "", text.strip())
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def decide(v, mode):
    """-> (accepted, category, conflict, summary). None accepted = unusable output."""
    if v is None:
        return None, None, False, None
    score = to_float(v.get("score"))
    if mode == "current":
        g, c, comp = v.get("grounded") is True, v.get("correct") is True, v.get("complete") is True
        if score is None:
            return None, None, False, None
        if score > 1.0:                                   # JudgeService normalization
            score /= 10.0
        accepted = g and c and score >= 0.70 and not (not comp and score < 0.75)
        all_true = g and c and comp
        conflict = (all_true and score < 0.5) or ((not g or not c) and score >= 0.8)
        summary = f"grounded={v.get('grounded')} correct={v.get('correct')} complete={v.get('complete')} score={score}"
        return accepted, None, conflict, summary
    verdict = str(v.get("verdict", "")).strip().upper()
    if verdict not in VERDICTS:
        return None, None, False, None
    accepted = verdict == "PASS"
    conflict = score is not None and ((accepted and score < 0.5) or (not accepted and score >= 0.8))
    return accepted, verdict, conflict, f"verdict={verdict} score={score}"


# ── deterministic escalation trigger ─────────────────────────────────────────────────────────────
# Skip digits glued to letters (model names like M3, CX-5 keep their digit after a letter or hyphen-letter).
NUM = re.compile(r"(?<![A-Za-z\d])\d[\d,]*(?:\.\d+)?")


def numbers(text):
    out = set()
    for m in NUM.findall(text):
        s = m.replace(",", "").rstrip(".")
        try:
            out.add(float(s))
        except ValueError:
            pass
    return out


def numbers_check(answer, context):
    """Every number in the answer must appear somewhere in the context. Returns the missing numbers."""
    ctx = numbers(" ".join(context))
    return sorted(n for n in numbers(answer) if n not in ctx)


def fmt(n):
    return str(int(n)) if n == int(n) else str(n)


def main():
    p = argparse.ArgumentParser(description="Nemotron-30B vs gpt-oss-120b as RAG judge")
    p.add_argument("--prompt", default="verdict", choices=["current", "verdict"], help="judge contract to test")
    p.add_argument("--runs", type=int, default=1, help="repeat each case N times")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high"], help="gpt-oss reasoning effort")
    p.add_argument("--nemotron-think", action="store_true", help="turn on Nemotron reasoning")
    p.add_argument("--temperature", type=float, default=0.1, help="same for both (app judge uses 0.1)")
    p.add_argument("--max-tokens", type=int, default=1024, help="same for both (app judge uses 1024)")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--only", help="run a single case by name")
    p.add_argument("--models", default="nemotron,gptoss", help="comma list: nemotron,gptoss")
    p.add_argument("-v", "--verbose", action="store_true", help="print raw model output")
    args = p.parse_args()

    all_models = {"nemotron": ("nemotron-30b", call_nemotron), "gptoss": ("gpt-oss-120b", call_gptoss)}
    models = [all_models[m.strip()] for m in args.models.split(",") if m.strip() in all_models]
    template = PROMPT_CURRENT if args.prompt == "current" else PROMPT_VERDICT
    system = SYSTEM.format(effort=args.effort)
    cases = [c for c in CASES if not args.only or c[0] == args.only]
    totals = {m: {"dec": 0, "cat": 0, "cat_n": 0, "conflict": 0, "json_ok": 0, "calls": 0, "ms": 0}
              for m, _ in models}
    trig = {"bad_caught": 0, "bad": 0, "good_flagged": 0, "good": 0}

    print(f"prompt={args.prompt} | gpt-oss effort={args.effort} | nemotron thinking="
          f"{'on' if args.nemotron_think else 'off'} | temperature={args.temperature} | "
          f"max_tokens={args.max_tokens} | runs={args.runs}\n")

    for name, skill, question, answer, context, should_accept, ok_verdicts in cases:
        missing = numbers_check(answer, context)
        fired = bool(missing)
        if should_accept:
            trig["good"] += 1
            trig["good_flagged"] += fired
        else:
            trig["bad"] += 1
            trig["bad_caught"] += fired
        want = "ACCEPT" if should_accept else "REJECT"
        if args.prompt == "verdict" and not should_accept:
            want += " (" + "/".join(sorted(ok_verdicts)) + ")"
        trig_txt = (f"FIRES, not in context: {', '.join(fmt(n) for n in missing)}" if fired else "passes")
        print(f"== {name}: {skill}\n   expected: {want}   | numbers check: {trig_txt}")

        user = user_message(template, question, answer, context)
        for model, fn in models:
            for run in range(args.runs):
                t0 = time.perf_counter()
                try:
                    raw, finish = fn(system, user, args)
                    err = None
                except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
                    raw, finish, err = "", None, str(e)
                ms = int((time.perf_counter() - t0) * 1000)
                v = extract_json(raw)
                accepted, category, conflict, summary = decide(v, args.prompt)
                t = totals[model]
                t["calls"] += 1
                t["ms"] += ms
                t["json_ok"] += accepted is not None
                t["conflict"] += conflict
                dec_ok = accepted is not None and accepted == should_accept
                t["dec"] += dec_ok
                if args.prompt == "verdict" and not should_accept and accepted is False:
                    t["cat_n"] += 1
                    t["cat"] += category in ok_verdicts
                mark = "ERR " if err else ("OK  " if dec_ok else "MISS")
                tag = f" run{run + 1}" if args.runs > 1 else ""
                if err:
                    detail = f"error: {err}"
                elif accepted is None:
                    detail = f"unusable output (finish_reason={finish}): {raw[:110]!r}"
                else:
                    detail = (f"{'ACCEPT' if accepted else 'REJECT'}  {summary}"
                              f"{'  [CONFLICT]' if conflict else ''} | {v.get('reason')}")
                print(f"   {mark} {model:<13}{tag} {ms:>6} ms  {detail}")
                if args.verbose:
                    print(f"        raw: {raw}")
        print()

    print("== Summary")
    verdict = args.prompt == "verdict"
    cols = ["model", "decision acc"] + (["category acc"] if verdict else []) + ["conflicts", "valid JSON", "avg latency"]
    widths = [14, 16] + ([14] if verdict else []) + [11, 12, 13]
    print("   " + "".join(c.ljust(w) for c, w in zip(cols, widths)))
    for model, _ in models:
        t = totals[model]
        n = max(t["calls"], 1)
        vals = [model, f"{t['dec']}/{t['calls']} ({100 * t['dec'] / n:.0f}%)"]
        if verdict:
            vals.append(f"{t['cat']}/{t['cat_n']}")
        vals += [f"{t['conflict']}/{t['calls']}", f"{t['json_ok']}/{t['calls']}", f"{t['ms'] // n} ms"]
        print("   " + "".join(v.ljust(w) for v, w in zip(vals, widths)))
    print(f"\n   numbers check: caught {trig['bad_caught']}/{trig['bad']} bad answers, "
          f"false alarm on {trig['good_flagged']}/{trig['good']} good answers")


if __name__ == "__main__":
    main()
