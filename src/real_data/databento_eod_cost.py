import argparse
import csv
import time as time_mod
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import databento as db
from dotenv import load_dotenv

load_dotenv()

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

DATASET = "OPRA.PILLAR"
SCHEMA = "cbbo-1m"
SYMBOLS = ["SPXW.OPT"]

# 5-min EOD window ending at the 16:15 SPXW close (a couple extra minutes for redundancy)
EOD_START = time(16, 10)
EOD_END = time(16, 15)
WINDOW = "eod"  # manifest cost key: (date, window, schema) -> get_cost is per-window/schema

SLEEP_S = 2.0  # be polite to the metadata endpoint between requests

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "datasets" / "raw" / "spxw" / "manifest.csv"
FIELDS = ["date", "window", "schema", "status", "est_cost", "size_bytes", "pulled_at"]


def load_manifest():
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, newline="") as f:
        return {(r["date"], r["window"], r["schema"]): r for r in csv.DictReader(f)}


def write_manifest(rows):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])


client = db.Historical()

p = argparse.ArgumentParser()
p.add_argument("start", type=date.fromisoformat)
p.add_argument("end", type=date.fromisoformat, nargs="?", default=date.today() - timedelta(days=1))
args = p.parse_args()

span = (args.end - args.start).days
days = [args.start + timedelta(days=i) for i in range(span + 1)]
days = [d for d in days if d.weekday() < 5]  # weekdays only; holidays just cost $0

manifest = load_manifest()
total = 0.0
priced = 0
for d in days:
    key = (d.isoformat(), WINDOW, SCHEMA)
    cached = manifest.get(key)
    if cached is not None:
        if cached["status"] == "closed":
            print(f"{d}  market closed (cached)")
        else:
            est = float(cached["est_cost"])
            total += est
            priced += 1
            print(f"{d}  {EOD_START:%H:%M}-{EOD_END:%H:%M} ET  estimated ${est:.4f}  (cached)")
        continue

    start = datetime.combine(d, EOD_START, tzinfo=ET).astimezone(UTC)
    end = datetime.combine(d, EOD_END, tzinfo=ET).astimezone(UTC)
    kw = dict(dataset=DATASET, symbols=SYMBOLS, stype_in="parent", schema=SCHEMA, start=start, end=end)
    row = dict(date=d.isoformat(), window=WINDOW, schema=SCHEMA,
               status="", est_cost="", size_bytes="", pulled_at="")
    try:
        est = client.metadata.get_cost(**kw)
    except db.BentoError:
        row["status"] = "closed"
        manifest[key] = row
        print(f"{d}  market closed")
        time_mod.sleep(SLEEP_S)
        continue
    row["status"] = "estimated"
    row["est_cost"] = f"{est:.6f}"
    manifest[key] = row
    total += est
    priced += 1
    print(f"{d}  {EOD_START:%H:%M}-{EOD_END:%H:%M} ET  estimated ${est:.4f}")
    time_mod.sleep(SLEEP_S)

write_manifest(manifest)

per_day = total / priced if priced else 0.0
print(f"\n{'trading days':<16}{priced}")
print(f"{'total cost':<16}${total:.4f}")
print(f"{'price per day':<16}${per_day:.4f}")
