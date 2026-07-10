import sys
import types

import numpy as np
import pandas as pd
import databento as db

if "_testcapi" not in sys.modules:  # vollib's LBR backend imports DBL_MIN/MAX from it; not always built
    _shim = types.ModuleType("_testcapi")
    _shim.DBL_MIN, _shim.DBL_MAX = sys.float_info.min, sys.float_info.max
    sys.modules["_testcapi"] = _shim

from vollib.black.implied_volatility import implied_volatility as black_iv


def load(path, trade_day):
    df = db.DBNStore.from_file(path).to_df().reset_index()
    sym = df["symbol"].str                              # OSI: root(6) YYMMDD C/P strike*1000(8)
    df["expiry"] = pd.to_datetime(sym[6:12], format="%y%m%d")
    df["cp"] = sym[12].str.lower()
    df["strike"] = sym[13:21].astype(int) / 1000
    df["dte"] = (df["expiry"] - pd.Timestamp(trade_day)).dt.days
    df["tau"] = df["dte"] / 365
    df = df.rename(columns={"bid_px_00": "bid", "ask_px_00": "ask",
                            "bid_sz_00": "bid_sz", "ask_sz_00": "ask_sz"})
    df["mid"] = (df["bid"] + df["ask"]) / 2
    return df[["ts_recv", "expiry", "cp", "strike", "dte", "tau",
               "bid", "ask", "mid", "bid_sz", "ask_sz"]]


def clean(df, min_mid=0.10):
    ok = (df["bid"] > 0) & (df["ask"] >= df["bid"]) & (df["mid"] >= min_mid)
    return df[ok].copy()


def _fit_forward(calls, puts, atm_band):
    strikes = calls.index.intersection(puts.index)
    if len(strikes) < 3:
        return np.nan, np.nan
    K = strikes.values.astype(float)
    diff = (calls[strikes] - puts[strikes]).values

    def line(K, diff):
        intercept, slope = np.linalg.lstsq(np.c_[np.ones_like(K), -K], diff, rcond=None)[0]
        return intercept / slope, slope

    F, disc = line(K, diff)
    near = np.abs(np.log(K / F)) < atm_band                 # refit where quotes are tight
    if near.sum() >= 3:
        F, disc = line(K[near], diff[near])
    return F, disc


def forwards(df, atm_band=0.10):
    # C_K - P_K = e^{-rt}(F - K): the slope is the discount, intercept / discount is the forward.
    rows = []
    for (ts, expiry), g in df.groupby(["ts_recv", "expiry"], sort=False):
        calls = g[g.cp == "c"].set_index("strike")["mid"]
        puts = g[g.cp == "p"].set_index("strike")["mid"]
        F, disc = _fit_forward(calls, puts, atm_band)
        tau = g["tau"].iloc[0]
        r = -np.log(disc) / tau if disc > 0 and tau > 0 else np.nan
        rows.append((ts, expiry, tau, F, disc, r))
    return pd.DataFrame(rows, columns=["ts_recv", "expiry", "tau", "F", "disc", "r"])


def _implied_vol(prices, F, K, r, tau, cp):
    out = np.full(len(prices), np.nan)
    for i in range(len(prices)):
        try:
            out[i] = black_iv(prices[i], F[i], K[i], r[i], tau[i], cp[i])
        except Exception:                                   # below intrinsic / above max / no convergence
            pass
    return out


def invert(df, fwd, otm_only=True):
    q = df.merge(fwd[["ts_recv", "expiry", "F", "disc", "r"]], on=["ts_recv", "expiry"])
    q = q[q["F"].notna() & (q["tau"] > 0)]
    if otm_only:
        q = q[((q.cp == "p") & (q.strike < q.F)) | ((q.cp == "c") & (q.strike > q.F))]
    q = q.copy()
    q["k"] = np.log(q["strike"] / q["F"])
    q["z"] = q["k"] / np.sqrt(q["tau"])

    F, K, r, tau, cp = (q[c].values for c in ["F", "strike", "r", "tau", "cp"])
    q["bid_iv"] = _implied_vol(q["bid"].values, F, K, r, tau, cp)
    q["ask_iv"] = _implied_vol(q["ask"].values, F, K, r, tau, cp)
    q["mid_iv"] = (q["bid_iv"] + q["ask_iv"]) / 2

    q = q.dropna(subset=["bid_iv", "ask_iv"])
    return q[q["ask_iv"] >= q["bid_iv"]]                     # drop the rare numerically-crossed IVs


def build(path, trade_day, min_mid=0.10, atm_band=0.10, otm_only=True):
    df = clean(load(path, trade_day), min_mid)
    fwd = forwards(df, atm_band)
    return invert(df, fwd, otm_only), fwd
