from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_generation.data_preperation import sample_context_sizes
from src.data_generation.grid import sample_arb_grid
from src.data_generation.noise import BID, ASK, TRUE

REPO = Path(__file__).resolve().parents[2]
PATH = REPO / "datasets" / "processed" / "spxw.parquet"


def load_surfaces(cfg=None):
    df = pd.read_parquet(PATH)
    if cfg is not None:
        z, t = cfg["z"], cfg["ttm"]
        df = df[df["z"].between(z["min"], z["max"]) & df["tau"].between(t["min"], t["max"])]
    return [g for _, g in df.groupby("date", sort=True)]


def temporal_split(start, end=None, val_months=1, test_months=3, cfg=None):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end or date.today() - timedelta(days=1))
    pool = [s for s in load_surfaces(cfg) if start <= s["date"].iloc[0] <= end]

    last = pool[-1]["date"].iloc[0]
    test_start = last - pd.DateOffset(months=test_months)
    val_start = test_start - pd.DateOffset(months=val_months)
    day = lambda s: s["date"].iloc[0]

    train = [s for s in pool if day(s) < val_start]
    val = [s for s in pool if val_start <= day(s) < test_start]
    test = [s for s in pool if day(s) >= test_start]
    return train, val, test


def _split_surface(s, nc, arb_rows=None, n_heldout=None):
    z, tau = s["z"].values, s["tau"].values
    bid, ask = s["bid_iv"].values, s["ask_iv"].values

    w = np.exp(-0.5 * (z / 0.25) ** 2)                    # ATM-weighted in z (z=0 is ATM)
    w /= w.sum()
    ctx = np.random.choice(len(z), size=nc, replace=False, p=w)
    held = np.setdiff1d(np.arange(len(z)), ctx)
    if n_heldout is not None and n_heldout < len(held):   # fixed count -> equal query lengths -> groupable
        held = np.random.choice(held, size=n_heldout, replace=False)

    X_train = np.column_stack([np.tile(z[ctx], 2), np.tile(tau[ctx], 2), np.repeat([BID, ASK], nc)])
    y_train = np.concatenate([bid[ctx], ask[ctx]])

    held_rows = np.column_stack([z[held], tau[held], np.full(len(held), TRUE)])
    y_held = np.column_stack([bid[held], ask[held]])
    if arb_rows is None:
        return (X_train, y_train), (held_rows, y_held)

    X_test = np.vstack([arb_rows, held_rows])             # arb grid first, then held-out quotes
    y_test = np.full((len(X_test), 2), np.nan)
    y_test[len(arb_rows):] = y_held
    return (X_train, y_train), (X_test, y_test)


def build_task(pool, n, n_context, cfg=None, n_heldout=None, size_group=1):
    sizes = sample_context_sizes(n_context, n, group=size_group)
    train, test = [], []
    for start in range(0, n, size_group):
        arb_rows = sample_arb_grid(cfg) if cfg is not None else None
        for nc in sizes[start:start + size_group]:
            tr, te = _split_surface(pool[np.random.randint(len(pool))], nc, arb_rows, n_heldout)
            train.append(tr)
            test.append(te)
    return train, test


def make_real_eval_set(pool, sizes, cfg=None, n_heldout=None):
    train, test = [], []
    for size in sizes:
        arb_rows = sample_arb_grid(cfg) if cfg is not None else None
        for s in pool:
            tr, te = _split_surface(s, size, arb_rows, n_heldout)
            train.append(tr)
            test.append(te)
    return train, test
