import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.real_data import quotes

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "datasets" / "raw" / "spxw"
OUT = REPO / "datasets" / "processed" / "spxw.parquet"

COLS = ["date", "expiry", "tau", "strike", "z", "bid_iv", "ask_iv", "mid_iv"]


def surface(day):
    q, _ = quotes.build(RAW / f"{day}.dbn.zst", day)
    q = q[q.ts_recv == q.ts_recv.max()].copy()
    q["date"] = pd.Timestamp(day)
    return q[COLS]


p = argparse.ArgumentParser()
p.add_argument("start", type=date.fromisoformat)
p.add_argument("end", type=date.fromisoformat, nargs="?", default=date.today() - timedelta(days=1))
args = p.parse_args()

days = [args.start + timedelta(days=i) for i in range((args.end - args.start).days + 1)]
days = [d.isoformat() for d in days if d.weekday() < 5]

pulled = {f.name.split(".")[0] for f in RAW.glob("*.dbn.zst")}

old = pd.read_parquet(OUT) if OUT.exists() else None
done = set(old["date"].dt.strftime("%Y-%m-%d")) if old is not None else set()

new = []
missing = skipped = failed = 0
for d in days:
    if d in done:
        skipped += 1
        continue
    if d not in pulled:
        missing += 1
        print(f"{d}  no raw file")
        continue
    try:
        s = surface(d)
    except Exception as e:
        failed += 1
        print(f"{d}  failed: {e}")
        continue
    new.append(s)
    print(f"{d}  {len(s)} quotes")

if new:
    frames = new if old is None else [old, *new]
    out = pd.concat(frames, ignore_index=True).sort_values(["date", "expiry", "strike"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
else:
    out = old

print(f"\nprocessed {len(new)}   skipped {skipped}   missing {missing}   failed {failed}")
if out is not None:
    print(f"surfaces {out['date'].nunique()}   rows {len(out)}")
