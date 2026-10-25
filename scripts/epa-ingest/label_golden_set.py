#!/usr/bin/env python3
"""
label_golden_set.py — make the vehicle golden set valid for an open-world fleet.

The original 29 entries were written when only ~8 curated vehicles existed, so
expectedVehicleIds assumed those were the ONLY right answers. With thousands of
EPA vehicles, "electric vehicles" correctly returning a Chevy Equinox EV was
scored as a miss.

For attribute queries that real EPA/NHTSA data can answer, this script:
  * marks the entry "openWorld": true
  * fills acceptableVehicleIds with every ingested vehicle that satisfies a
    rule derived from real EPA/NHTSA fields (rules below, one per entry)
  * sets minimumRecallCount = min(5, number of relevant vehicles)

Entries whose answer depends on data EPA does not have (horsepower, torque,
MSRP, resale, maintenance, rankings) or that are subjective stay closed-world
and unchanged. Their low scores at scale are real retrieval problems, not
labeling artifacts.

Re-runnable: it always starts from expectedVehicleIds plus the ORIGINAL
hand-written acceptables (kept in "curatedAcceptableVehicleIds"), so running it
twice gives the same file.

Usage (from C:\\dev\\ai-rag-assistant):
    python scripts\\epa-ingest\\label_golden_set.py
    python scripts\\epa-ingest\\label_golden_set.py --dry-run
"""

import argparse
import json
import re
from pathlib import Path

GOLDEN = "src/main/resources/eval/vehicle_golden_set.json"
VEHICLES = "data/epa/vehicles_enriched.jsonl"
OPEN_WORLD_MIN = 5


def combined_mpg(v):
    """EPA combined MPG/MPGe from the features list prepare_epa.py writes."""
    for f in v.get("features") or []:
        m = re.search(r"EPA combined fuel economy (\d+) (MPGe|MPG)", f)
        if m:
            return int(m.group(1))
    return None


def annual_fuel_cost(v):
    for f in v.get("features") or []:
        m = re.search(r"annual fuel cost \$([\d,]+)", f)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def is_electric(v):
    return v.get("fuelType") == "electric"


ALL_WHEEL = {"AWD", "4WD/AWD", "4WD"}

# entryId -> (rule, human-readable rule description written into "labelRule")
RULES = {
    "S4-003": (is_electric,
               "fuelType == electric"),
    "S4-004": (lambda v: not is_electric(v) and (combined_mpg(v) or 0) >= 40,
               "non-EV with EPA combined MPG >= 40"),
    "S4-005": (lambda v: v.get("drivetrain") in ALL_WHEEL,
               "drivetrain in {AWD, 4WD/AWD, 4WD}"),
    "S2-002": (lambda v: (v.get("bodyStyle") in {"SUV", "Minivan", "Wagon"}
                          or "midsize" in (v.get("vehicleClass") or "")
                          or "large cars" in (v.get("vehicleClass") or ""))
                         and (annual_fuel_cost(v) or 10**9) <= 1300,
               "family body/class (SUV, minivan, wagon, midsize or large car) "
               "with EPA annual fuel cost <= $1,300"),
    "S2-006": (lambda v: v.get("bodyStyle") == "Pickup Truck"
                         and v.get("drivetrain") in {"4WD", "Part-time 4WD", "AWD", "4WD/AWD"},
               "pickup truck with 4WD/AWD"),
    "S3-002": (lambda v: (is_electric(v) and (v.get("electricRangeMiles") or 0) >= 250)
                         or (not is_electric(v) and (combined_mpg(v) or 0) >= 45),
               "EV with EPA range >= 250 mi, or non-EV with combined MPG >= 45"),
    "EDGE-001": (lambda v: bool(v.get("recalls")),
                 "has at least one NHTSA recall attached"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", default=GOLDEN)
    ap.add_argument("--vehicles", default=VEHICLES)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    vehicles = [json.loads(l) for l in open(args.vehicles, encoding="utf-8") if l.strip()]
    entries = json.load(open(args.golden, encoding="utf-8"))

    for e in entries:
        # Preserve the hand-written acceptables once, so re-runs are idempotent.
        if "curatedAcceptableVehicleIds" not in e:
            e["curatedAcceptableVehicleIds"] = list(e.get("acceptableVehicleIds") or [])
        curated_acc = e["curatedAcceptableVehicleIds"]

        if e["entryId"] not in RULES:
            e["openWorld"] = False
            e["acceptableVehicleIds"] = curated_acc
            e.pop("labelRule", None)
            continue

        rule, desc = RULES[e["entryId"]]
        expected = set(e["expectedVehicleIds"])
        generated = sorted(v["vehicleId"] for v in vehicles if rule(v) and v["vehicleId"] not in expected)
        acceptable = sorted(set(curated_acc) | set(generated))
        relevant = len(expected | set(acceptable))

        e["openWorld"] = True
        e["acceptableVehicleIds"] = acceptable
        e["labelRule"] = desc
        e["minimumRecallCount"] = min(OPEN_WORLD_MIN, relevant)
        print(f"{e['entryId']:9} open-world  relevant={relevant:5}  min={e['minimumRecallCount']}  rule: {desc}")

    closed = [e["entryId"] for e in entries if not e["openWorld"]]
    print(f"\n{len(entries) - len(closed)} open-world entries, {len(closed)} closed-world (unchanged): {', '.join(closed)}")

    if args.dry_run:
        print("\n--dry-run: golden set not written")
        return
    with open(args.golden, "w", encoding="utf-8", newline="\n") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nWrote {args.golden}")


if __name__ == "__main__":
    main()
