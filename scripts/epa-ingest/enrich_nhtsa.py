#!/usr/bin/env python3
"""
enrich_nhtsa.py — OPTIONAL. Adds real NHTSA crash-test ratings and recalls to
the vehicles produced by prepare_epa.py. Fills the SafetyRecord and
RecallRecord parts of RichVehicleRecord, which feed chunk 5 (safety) and
chunks 20+ (one per recall).

Standard library only. Free public NHTSA APIs, no key needed.
Responses are cached per (year, make, model) under data/epa/nhtsa_cache/,
so re-runs are fast and you can stop/restart at any time.

Usage (from C:\\dev\\ai-rag-assistant):
    python scripts\\epa-ingest\\enrich_nhtsa.py
    python scripts\\epa-ingest\\enrich_nhtsa.py --max-recalls 3 --delay 0.25

Input : data/epa/vehicles.jsonl
Output: data/epa/vehicles_enriched.jsonl   (ingest_bulk.py prefers this if present)

Matching note: EPA and NHTSA spell model names differently ("F150" vs "F-150").
The script tries a few spellings per model and reports the match rate.
Vehicles without a match are kept as-is, with no invented safety data.
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

API = "https://api.nhtsa.gov"


class Nhtsa:
    def __init__(self, cache_dir: Path, delay: float):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.calls = 0

    def get(self, path: str):
        key = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_")[:180]
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        url = API + path
        data = {"Results": [], "results": []}
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ai-rag-assistant-ingest/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code in (400, 404):
                    break                      # no data for this vehicle; cache the empty result
                time.sleep(2 ** attempt)
            except Exception:
                time.sleep(2 ** attempt)
        else:
            return {"Results": [], "results": []}   # transient failure: don't cache
        self.calls += 1
        time.sleep(self.delay)
        cached.write_text(json.dumps(data), encoding="utf-8")
        return data


def model_candidates(model: str):
    """'F150' -> ['F150', 'F-150', 'F 150'];  'CR-V' -> ['CR-V', 'CRV'];  'Model 3' -> ['Model 3']."""
    m = model.strip()
    cands = [m]
    hyph = re.sub(r"(?<=[A-Za-z])(?=\d)", "-", m)
    if hyph != m:
        cands += [hyph, hyph.replace("-", " ")]
    if "-" in m:
        cands.append(m.replace("-", ""))
    # No first-word fallback: "Mustang Mach-E" -> "Mustang" would attach a
    # different vehicle's ratings. A missing rating beats a wrong one.
    seen, out = set(), []
    for c in cands:
        if c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


def stars(v):
    if v in (None, "", "Not Rated"):
        return None
    return f"{v}-star"


def fetch_safety(api: Nhtsa, year, make, model):
    q = lambda s: urllib.parse.quote(str(s))
    for cand in model_candidates(model):
        listing = api.get(f"/SafetyRatings/modelyear/{year}/make/{q(make)}/model/{q(cand)}")
        results = listing.get("Results") or []
        if not results:
            continue
        vid = results[0].get("VehicleId")
        if not vid:
            continue
        detail = (api.get(f"/SafetyRatings/VehicleId/{vid}").get("Results") or [{}])[0]

        def std(*keys):
            for k, v in detail.items():
                if any(key.lower() in k.lower() for key in keys):
                    return str(v).strip().lower() == "standard"
            return False

        overall = stars(detail.get("OverallRating"))
        if not overall:
            continue
        return {
            "nhtsaOverall": overall,
            "nhtsaFrontal": stars(detail.get("OverallFrontCrashRating")),
            "nhtsaSide": stars(detail.get("OverallSideCrashRating")),
            "nhtsaRollover": stars(detail.get("RolloverRating")),
            "iihsOverall": None,           # IIHS data is not in the NHTSA API
            "iihsSmallOverlap": None,
            "iihsHeadlights": None,
            "hasAutomaticEmergencyBraking": std("CrashImminentBraking", "AutomaticEmergencyBraking"),
            "hasLaneDepartureWarning": std("LaneDepartureWarning"),
            "hasBlindSpotMonitoring": std("BlindSpot"),
        }
    return None


def nhtsa_date(s):
    """NHTSA returns dates like '21/03/2024' (dd/mm/yyyy); normalize to ISO."""
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except (TypeError, ValueError):
            pass
    return s or "unknown date"


def fetch_recalls(api: Nhtsa, year, make, model, max_recalls):
    q = urllib.parse.quote
    for cand in model_candidates(model):
        data = api.get(f"/recalls/recallsByVehicle?make={q(make)}&model={q(cand)}&modelYear={year}")
        results = data.get("results") or data.get("Results") or []
        if not results:
            continue
        recs = []
        for r in results:
            remedy = (r.get("Remedy") or "").strip()
            recs.append({
                "recallId": r.get("NHTSACampaignNumber") or "unknown",
                "date": nhtsa_date(r.get("ReportReceivedDate")),
                "component": (r.get("Component") or "Unspecified").title(),
                "description": " ".join(filter(None, [(r.get("Summary") or "").strip(),
                                                      (r.get("Consequence") or "").strip()]))[:900],
                "remedy": remedy[:400] or "See NHTSA campaign",
                "remedyAvailable": bool(remedy),
            })
        recs.sort(key=lambda x: x["date"], reverse=True)
        return recs[:max_recalls]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="data/epa/vehicles.jsonl")
    ap.add_argument("--out", default="data/epa/vehicles_enriched.jsonl")
    ap.add_argument("--cache-dir", default="data/epa/nhtsa_cache")
    ap.add_argument("--max-recalls", type=int, default=5,
                    help="recalls kept per vehicle (each becomes its own chunk)")
    ap.add_argument("--delay", type=float, default=0.2, help="seconds between uncached API calls")
    ap.add_argument("--no-recalls", action="store_true")
    ap.add_argument("--no-safety", action="store_true")
    args = ap.parse_args()

    api = Nhtsa(Path(args.cache_dir), args.delay)
    vehicles = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    lookup = {}   # (year, make, model) -> (safety, recalls); many EPA variants share one model
    safety_hits = recall_hits = 0
    t0 = time.time()

    with open(args.out, "w", encoding="utf-8") as out:
        for i, v in enumerate(vehicles, 1):
            key = (v["year"], v["make"], v["model"])
            if key not in lookup:
                safety = None if args.no_safety else fetch_safety(api, *key)
                recalls = [] if args.no_recalls else fetch_recalls(api, *key, args.max_recalls)
                lookup[key] = (safety, recalls)
            safety, recalls = lookup[key]
            if safety:
                v["safety"] = safety
                safety_hits += 1
            if recalls:
                v["recalls"] = recalls
                recall_hits += 1
            out.write(json.dumps(v, ensure_ascii=False) + "\n")
            if i % 250 == 0 or i == len(vehicles):
                print(f"  {i}/{len(vehicles)} vehicles | {len(lookup)} models looked up | "
                      f"{api.calls} API calls | {time.time() - t0:.0f}s")

    n = len(vehicles) or 1
    print(f"Done. Safety ratings matched: {safety_hits}/{len(vehicles)} ({100 * safety_hits / n:.0f}%). "
          f"Recalls attached: {recall_hits}/{len(vehicles)} ({100 * recall_hits / n:.0f}%).")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
