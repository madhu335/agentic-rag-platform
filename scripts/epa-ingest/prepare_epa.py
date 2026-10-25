#!/usr/bin/env python3
"""
prepare_epa.py — download the EPA fuel economy dataset and convert it into
RichVehicleRecord JSON for POST /vehicles/ingest/bulk.

Standard library only (no pip installs).

Usage (from C:\\dev\\ai-rag-assistant):
    python scripts\\epa-ingest\\prepare_epa.py --from-year 2015 --to-year 2025
    python scripts\\epa-ingest\\prepare_epa.py --from-year 2024 --to-year 2025 --limit 500

Output:
    data\\epa\\vehicles.csv              (cached download)
    data\\epa\\vehicles.jsonl            (one RichVehicleRecord per line)

Design notes
------------
* vehicleId format:  <make>-<basemodel>-<year>-<variant>-epa<id>
  e.g. "ford-f150-2024-pickup-4wd-3-5l-v6-turbo-epa47512"
  It MUST match PgVectorStore.inferDocType()'s vehicle pattern  .*-\\d{4}(-.*)?$
  or the chunks silently land in the 'pdf' partition and become invisible to
  fleet search. Every generated ID is validated against that regex.
* The "epa<id>" suffix makes IDs unique (EPA has many rows per make/model/year:
  2WD/4WD, engines, transmissions) and keeps them from colliding with the
  hand-curated golden-set vehicles.
* Only real EPA fields are used. EPA has no horsepower, torque, MSRP, reviews,
  rankings or maintenance data, so those stay null rather than being invented.
"""

import argparse
import csv
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

EPA_URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip"

# Same rule as PgVectorStore.inferDocType() — keep these in sync.
VEHICLE_ID_PATTERN = re.compile(r".*-\d{4}(-.*)?$")

# vehicle_summaries column limits (V2__partition_two_tier_paradedb.sql)
TRIM_MAX = 100


# ─── Helpers ──────────────────────────────────────────────────────────────────

def slug(text: str) -> str:
    text = (text or "").lower().replace("&", " and ").replace("+", " plus ")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-")


def col(row: dict, *names, default=""):
    """Case-insensitive column lookup with fallbacks (EPA header casing varies)."""
    lower = {k.lower(): v for k, v in row.items()}
    for n in names:
        v = lower.get(n.lower())
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


def to_int(v):
    try:
        f = float(v)
        return int(round(f)) if f > 0 else None
    except (TypeError, ValueError):
        return None


def to_float(v):
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


# ─── Download ────────────────────────────────────────────────────────────────

def download_csv(cache: Path, refresh: bool) -> Path:
    if cache.exists() and not refresh:
        print(f"Using cached {cache}")
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {EPA_URL} ...")
    req = urllib.request.Request(EPA_URL, headers={"User-Agent": "ai-rag-assistant-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        cache.write_bytes(zf.read(name))
    print(f"Saved {cache} ({cache.stat().st_size / 1e6:.1f} MB)")
    return cache


# ─── Field mapping ───────────────────────────────────────────────────────────

def fuel_type(row) -> str:
    atv = col(row, "atvType", "atvtype").lower()
    fuel1 = col(row, "fuelType1", "fuelType").lower()
    if atv == "ev":
        return "electric"
    if atv == "plug-in hybrid":
        return "plug-in hybrid"
    if atv == "hybrid":
        return "hybrid"
    if atv == "diesel" or "diesel" in fuel1:
        return "diesel"
    if atv.startswith("ffv"):
        return "flex-fuel (gasoline or E85)"
    if atv.startswith("cng") or atv.startswith("bifuel"):
        return atv
    if "premium" in fuel1:
        return "premium gasoline"
    return "gasoline"


def engine_desc(row, fuel: str) -> str:
    if fuel == "electric":
        motor = col(row, "evMotor")
        return f"Electric motor ({motor})" if motor else "Electric motor"
    parts = []
    displ = to_float(col(row, "displ"))
    cyl = to_int(col(row, "cylinders"))
    if displ:
        parts.append(f"{displ:.1f}L")
    if cyl:
        parts.append(f"{cyl}-cylinder")   # EPA doesn't say inline/V/flat, so don't guess
    if col(row, "tCharger") and col(row, "sCharger"):
        parts.append("turbocharged and supercharged")
    elif col(row, "tCharger"):
        parts.append("turbocharged")
    elif col(row, "sCharger"):
        parts.append("supercharged")
    if fuel in ("hybrid", "plug-in hybrid"):
        motor = col(row, "evMotor")
        parts.append(f"{fuel} ({motor})" if motor else fuel)
    return " ".join(parts) if parts else "N/A"


def drivetrain(row) -> str:
    d = col(row, "drive").lower()
    if "front" in d:
        return "FWD"
    if "rear" in d:
        return "RWD"
    if "all-wheel" in d and "4-wheel" not in d:
        return "AWD"
    if "part-time" in d:
        return "Part-time 4WD"
    if "4-wheel or all-wheel" in d:
        return "4WD/AWD"
    if "4-wheel" in d:
        return "4WD"
    if "2-wheel" in d:
        return "2WD"
    return col(row, "drive") or "N/A"


def transmission(row, fuel: str = "") -> str:
    if fuel == "electric":
        return "single-speed direct drive"
    t = col(row, "trany")
    if not t:
        return "N/A"
    m = re.search(r"(\d+)", t)
    speeds = f"{m.group(1)}-speed " if m else ""
    if "variable" in t.lower() or "(av" in t.lower() or "cvt" in t.lower():
        return "CVT (continuously variable)"
    if t.lower().startswith("auto"):
        return f"{speeds}automatic".strip()
    if t.lower().startswith("manual"):
        return f"{speeds}manual".strip()
    return t


def body_and_class(row):
    vclass = col(row, "VClass")
    v = vclass.lower()
    if "pickup" in v:
        body = "Pickup Truck"
    elif "sport utility" in v:
        body = "SUV"
    elif "minivan" in v:
        body = "Minivan"
    elif "van" in v:
        body = "Van"
    elif "station wagon" in v:
        body = "Wagon"
    elif "two seater" in v:
        body = "Two-seater"
    else:
        body = "Car"
    cls = re.sub(r"\s+(2wd|4wd|awd)$", "", v).strip()
    cls = cls.replace("standard pickup trucks", "full-size pickup truck") \
             .replace("small pickup trucks", "mid-size pickup truck") \
             .replace("sport utility vehicle", "SUV")
    return body, (cls or "N/A")


def variant_label(row, model: str, base: str, drive: str, engine: str) -> str:
    """Human-readable trim/variant: whatever EPA's model name adds beyond the base model."""
    extra = model
    if base and model.lower().startswith(base.lower()):
        extra = model[len(base):]
    extra = extra.strip(" -/")
    # Skip the drive label when EPA's model name already states one ("Pickup 2WD" + "RWD").
    has_drive = re.search(r"\b(2wd|4wd|awd|fwd|rwd|4x4|4matic|xdrive|quattro)\b", extra.lower())
    bits = [b for b in (extra, "" if has_drive else drive, engine) if b and b != "N/A"]
    label = " ".join(bits) if bits else "base"
    # vehicle_summaries.trim is VARCHAR(100); a longer value makes the summary
    # upsert fail and the vehicle disappears from Tier 1 fleet search.
    if len(label) > TRIM_MAX:
        label = label[:TRIM_MAX].rsplit(" ", 1)[0].rstrip(" (,")
    return label


def features(row, fuel: str):
    out = []
    fc = to_int(col(row, "fuelCost08"))
    if fc:
        out.append(f"EPA estimated annual fuel cost ${fc:,} (15,000 miles/year)")
    comb = to_int(col(row, "comb08"))
    if comb:
        unit = "MPGe" if fuel == "electric" else "MPG"
        out.append(f"EPA combined fuel economy {comb} {unit}")
    co2 = to_int(col(row, "co2TailpipeGpm"))
    if co2 is not None and fuel != "electric":
        out.append(f"Tailpipe CO2 {co2} grams per mile")
    if col(row, "startStop").upper() == "Y":
        out.append("engine stop-start system")
    if col(row, "tCharger"):
        out.append("turbocharged engine")
    if col(row, "sCharger"):
        out.append("supercharged engine")
    if col(row, "guzzler"):
        out.append("subject to gas guzzler tax")
    return out


def to_record(row):
    year = to_int(col(row, "year"))
    make = col(row, "make")
    model = col(row, "model")
    base = col(row, "baseModel", "basemodel") or model.split(" ")[0]
    epa_id = col(row, "id")
    if not (year and make and model and epa_id):
        return None

    fuel = fuel_type(row)
    engine = engine_desc(row, fuel)
    drive = drivetrain(row)
    body, vclass = body_and_class(row)
    trim = variant_label(row, model, base, drive, engine)

    vehicle_id = f"{slug(make)}-{slug(base)}-{year}-{slug(trim)}-epa{slug(epa_id)}"
    if vehicle_id.startswith("motortrend-") or not VEHICLE_ID_PATTERN.match(vehicle_id):
        return None

    rng = to_int(col(row, "range")) if fuel == "electric" else None

    return {
        "vehicleId": vehicle_id,
        "year": year,
        "make": make,
        "model": base,
        "trim": trim,
        "bodyStyle": body,
        "vehicleClass": vclass,
        "exteriorColor": None,
        "interiorColor": None,
        "engine": engine,
        "horsepower": None,       # not in EPA data
        "torque": None,           # not in EPA data
        "drivetrain": drive,
        "transmission": transmission(row, fuel),
        "zeroToSixty": None,
        "topSpeed": None,
        "fuelType": fuel,
        "mpgCity": col(row, "city08") or None,
        "mpgHighway": col(row, "highway08") or None,
        "electricRangeMiles": rng,
        "msrp": None,             # not in EPA data
        "invoicePrice": None,
        "destinationCharge": None,
        "features": features(row, fuel),
        "trimLevels": None,
        "rankings": None,
        "maintenanceSchedule": None,
        "ownershipCost": None,    # see VehicleChunkBuilder fix: chunk is skipped when null
        "safety": None,           # filled by enrich_nhtsa.py (optional)
        "recalls": None,          # filled by enrich_nhtsa.py (optional)
        "reviews": None,
        "summary": None,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-year", type=int, default=2015)
    ap.add_argument("--to-year", type=int, default=2025)
    ap.add_argument("--limit", type=int, default=0, help="max vehicles to write (0 = all)")
    ap.add_argument("--out-dir", default="data/epa")
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    csv_path = download_csv(out_dir / "vehicles.csv", args.refresh)
    out_path = out_dir / "vehicles.jsonl"

    seen, written, skipped = set(), 0, 0
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f, \
         open(out_path, "w", encoding="utf-8") as out:
        for row in csv.DictReader(f):
            year = to_int(col(row, "year"))
            if not year or year < args.from_year or year > args.to_year:
                continue
            rec = to_record(row)
            if rec is None or rec["vehicleId"] in seen:
                skipped += 1
                continue
            seen.add(rec["vehicleId"])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break

    print(f"Wrote {written} vehicles to {out_path} ({skipped} skipped: missing fields or bad ID)")
    if written == 0:
        print("No vehicles written — check --from-year/--to-year.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
