import sys
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

OPEN = time(9, 30)
CLOSE = time(16, 15)

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "datasets" / "raw" / "spxw"

day = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today() - timedelta(days=1)
start = datetime.combine(day, OPEN, tzinfo=ET).astimezone(UTC)
end = datetime.combine(day, CLOSE, tzinfo=ET).astimezone(UTC)

out = OUT_DIR / f"{day.isoformat()}.dbn.zst"
if out.exists():
    sys.exit(f"{out} already exists — not re-pulling.")

client = db.Historical()

kw = dict(dataset=DATASET, symbols=SYMBOLS, stype_in="parent", schema=SCHEMA, start=start, end=end)
est = client.metadata.get_cost(**kw)
if input(f"{day} full session — estimated cost ${est:.4f}. pull? [y/N] ").strip().lower() != "y":
    sys.exit("aborted.")

store = client.timeseries.get_range(**kw)

OUT_DIR.mkdir(parents=True, exist_ok=True)
store.to_file(out)

df = store.to_df()
print(f"saved {out}  ({out.stat().st_size / 1024**2:.1f} MiB on disk)")
print(f"rows: {len(df):,}   unique instruments: {df['symbol'].nunique():,}")
print(df.head())
