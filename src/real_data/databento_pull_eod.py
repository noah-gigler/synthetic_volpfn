import argparse
import csv
import sys
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
WINDOW = "eod"  # manifest cost key: (date, window, schema)

SLEEP_S = 2.0

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "datasets" / "raw" / "spxw"
MANIFEST = OUT_DIR / "manifest.csv"
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


p = argparse.ArgumentParser()
p.add_argument("start", type=date.fromisoformat)
p.add_argument("end", type=date.fromisoformat, nargs="?", default=date.today() - timedelta(days=1))
args = p.parse_args()

span = (args.end - args.start).days
days = [args.start + timedelta(days=i) for i in range(span + 1)]
days = [d for d in days if d.weekday() < 5]  # weekdays only

manifest = load_manifest()

have, to_pull, missing = [], [], []
for d in days:
    if (OUT_DIR / f"{d.isoformat()}.dbn.zst").exists():
        have.append(d)
        continue
    cached = manifest.get((d.isoformat(), WINDOW, SCHEMA))
    if cached is None:
        missing.append(d)
    elif cached["status"] == "closed":
        continue  # holiday, nothing to pull
    else:
        to_pull.append((d, float(cached["est_cost"])))

if have:
    print(f"{len(have)} dates already downloaded, skipping: {have[0]} .. {have[-1]}")

if missing:
    print(f"\nno cost estimate for {len(missing)} date(s): {missing[0]} .. {missing[-1]}")
    sys.exit("run databento_eod_cost.py over this range first.")

if not to_pull:
    sys.exit("nothing to pull.")

total = sum(c for _, c in to_pull)
print(f"\n{'days to pull':<16}{len(to_pull)}")
print(f"{'total cost':<16}${total:.4f}")
if input(f"\npull {len(to_pull)} days for ${total:.4f}? [y/N] ").strip().lower() != "y":
    sys.exit("aborted.")

client = db.Historical()
OUT_DIR.mkdir(parents=True, exist_ok=True)
for d, est in to_pull:
    start = datetime.combine(d, EOD_START, tzinfo=ET).astimezone(UTC)
    end = datetime.combine(d, EOD_END, tzinfo=ET).astimezone(UTC)
    kw = dict(dataset=DATASET, symbols=SYMBOLS, stype_in="parent", schema=SCHEMA, start=start, end=end)
    out = OUT_DIR / f"{d.isoformat()}.dbn.zst"

    store = client.timeseries.get_range(**kw)
    store.to_file(out)

    size = out.stat().st_size
    row = manifest[(d.isoformat(), WINDOW, SCHEMA)]
    row["status"] = "pulled"
    row["size_bytes"] = str(size)
    row["pulled_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    write_manifest(manifest)  # persist per-day so a crash mid-batch keeps progress
    print(f"{d}  saved {out.name}  ({size / 1024**2:.1f} MiB)")
    time_mod.sleep(SLEEP_S)

print(f"\npulled {len(to_pull)} days.")
