# Reproduces the OpDS benchmark (Wiedemann, Jacquier & Gonon, ICLR 2025, arXiv:2406.11520),
# Appendix C.5 / Table 5, which is itself Table 1 of Ackerer et al. (2020):
#
#   drop 50% of a surface's quotes, smooth from the retained half, then report MAPE at the
#   RETAINED points ("Train") and the DROPPED points ("Test"), as q05/q50/q95 over surfaces.
#
# Two drop settings, as in Ackerer:
#   interpolation - dropped points are scattered among retained ones (random 50%)
#   extrapolation - dropped points lie outside the retained region (outer moneyness band)
#
# Their numbers, Jan-Apr 2018 EOD SPX, MAPE in %:
#            Interpolation Train/Test        Extrapolation Train/Test
#   OpDS       0.5/0.7/1.0  0.5/0.7/1.1      0.5/0.7/1.0  0.7/0.9/1.3
#   DS         0.5/0.7/1.2  0.5/0.8/1.2      0.4/0.6/0.9  1.2/1.7/2.4
#
# Also reports their surface metrics d_abs (= MAPE) and d_spr = 2|BS(v_hat)-BS(v)|/(BS(ask)-BS(bid)),
# where d_spr <= 1 means the prediction prices inside the bid-ask spread.
# Their Table 1 (EOD SPX Jan 2021): <d_abs> 0.00272, <d_spr> 0.644.
#
# NOT matched, and not fixable here: they train on 49,089 intraday surfaces (2012-2021), we have
# ~1,000 EOD ones; they refit monthly; theirs is a GNO trained from scratch, ours a finetuned
# TabPFN. Their fit term is also vega-weighted and ours is not.
#
#   uv run python -m scripts.run_opds_benchmark <run-name> [final|best] [synthetic|real]
#
# The 3rd argument must match the --y-source the checkpoint was TRAINED with: real-data runs
# used "real", synthetic ones "synthetic". Getting it wrong silently produces garbage (the
# model reads inputs on the wrong scale) rather than an error.
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data_generation.noise import BID, ASK, TRUE, _bs_otm_price_vega
from scripts.run_finetuning import load_arb_grid
from src.evaluation.surface_eval import _get_eval_estimator, _predict_raw, eval_arbitrage_fine
from src.model.preprocessed_dataset import preprocess_surfaces
from src.real_data.dataloader import temporal_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 1
DROP_FRAC = 0.5          # Ackerer/OpDS protocol


def split_surface(s, mode, rng):
    """Return (retained_idx, dropped_idx) for one day, per the Ackerer drop settings."""
    z = s["z"].values
    n_drop = int(round(DROP_FRAC * len(z)))
    if mode == "interpolation":
        drop = rng.choice(len(z), size=n_drop, replace=False)
    else:
        # dropped points sit outside the retained region: take the widest-|z| half, so the
        # model must extend the smile rather than fill gaps inside it
        drop = np.argsort(-np.abs(z))[:n_drop]
    keep = np.setdiff1d(np.arange(len(z)), drop)
    return keep, drop


def metrics(pred, bid, ask, z, tau):
    mid = (bid + ask) / 2
    k = z * np.sqrt(tau)
    p_hat, _ = _bs_otm_price_vega(k, tau, pred)
    p_mid, _ = _bs_otm_price_vega(k, tau, mid)
    p_bid, _ = _bs_otm_price_vega(k, tau, bid)
    p_ask, _ = _bs_otm_price_vega(k, tau, ask)
    spread = p_ask - p_bid
    mape = np.abs(pred - mid) / mid
    ok = np.isfinite(spread) & (spread > 0) & np.isfinite(p_hat) & np.isfinite(p_mid)
    d_spr = 2 * np.abs(p_hat[ok] - p_mid[ok]) / spread[ok]
    inside = (pred >= bid) & (pred <= ask)
    return np.nanmean(mape) * 100, np.nanmean(d_spr), inside.mean()


def run(run_name, which="final", y_source="synthetic"):
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    sfx = "_real" if y_source == "real" else ""
    y_mean, y_scale = cfg["y_mean" + sfx], cfg["y_scale" + sfx]
    _, _, pool = temporal_split("2020-01-01", "2023-12-31", val_months=3, test_months=6, cfg=cfg)
    print(f"{run_name} ({which}.pt) | OpDS benchmark | y_source={y_source} | {len(pool)} EOD SPXW test days, "
          f"{int(np.median([len(s) for s in pool]))} quotes/day median, {DROP_FRAC:.0%} dropped")

    # run_name="pretrained" scores the un-finetuned TabPFN - the control that separates
    # "this config trains badly" from "finetuning at this scale actively damages the model"
    est, pretrained = _get_eval_estimator(None)
    state = pretrained if run_name == "pretrained" else torch.load(
        ROOT / "checkpoints" / run_name / f"{which}.pt",
        map_location="cuda" if torch.cuda.is_available() else "cpu")
    est.model_.load_state_dict(state)
    out = {}
    for mode in ("interpolation", "extrapolation"):
        rng = np.random.default_rng(SEED)
        train_list, queries, targets = [], [], []
        for s in pool:
            z, tau = s["z"].values, s["tau"].values
            bid, ask = s["bid_iv"].values, s["ask_iv"].values
            keep, drop = split_surface(s, mode, rng)
            nk = len(keep)
            # context = retained quotes as bid/ask row pairs (the schema the model was trained on)
            X_ctx = np.column_stack([np.tile(z[keep], 2), np.tile(tau[keep], 2),
                                     np.repeat([BID, ASK], nk)])
            y_ctx = np.concatenate([bid[keep], ask[keep]])
            idx = np.concatenate([keep, drop])          # score retained first, then dropped
            X_q = np.column_stack([z[idx], tau[idx], np.full(len(idx), TRUE)])
            train_list.append((X_ctx, y_ctx))
            queries.append((X_q, np.zeros(len(X_q))))
            targets.append((nk, bid[idx], ask[idx], z[idx], tau[idx]))

        surfaces = preprocess_surfaces(est, train_list, queries, np.random.default_rng(0),
                                       cfg["iv_max"], group_size=1,
                                       y_mean=y_mean, y_scale=y_scale)
        tr_m, te_m, te_spr, te_ins = [], [], [], []
        for (pred, _), (nk, bid, ask, z, tau) in zip(_predict_raw(est, surfaces), targets):
            tr_m.append(metrics(pred[:nk], bid[:nk], ask[:nk], z[:nk], tau[:nk])[0])
            m, sp, ins = metrics(pred[nk:], bid[nk:], ask[nk:], z[nk:], tau[nk:])
            te_m.append(m); te_spr.append(sp); te_ins.append(ins)
        q = lambda a: [float(np.percentile(a, p)) for p in (5, 50, 95)]
        # ARBITRAGE on the same contexts. Inside-spread is meaningless without it - any curve
        # threaded through the bid/ask band scores 100% inside. OpDS report L_but 6.4e-05,
        # L_cal 0.0 on SPX EOD, i.e. effectively arb-free.
        cf, md, wc, af = eval_arbitrage_fine(None, train_list, cfg, load_arb_grid(cfg),
                                             reload_state=state, group_size=1, iv_max=cfg["iv_max"],
                                             y_mean=y_mean, y_scale=y_scale)
        out[mode] = dict(train_mape=q(tr_m), test_mape=q(te_m),
                         test_d_spr=float(np.mean(te_spr)), test_inside=float(np.mean(te_ins)),
                         cell_frac=cf, mean_depth=md, worst_cell=wc, arb_free=af)

    ref = {"interpolation": ("0.5/0.7/1.0", "0.5/0.7/1.1"), "extrapolation": ("0.5/0.7/1.0", "0.7/0.9/1.3")}
    print(f"\n{'':16} {'Train q05/q50/q95':>22} {'Test q05/q50/q95':>22}   MAPE %")
    for mode, r in out.items():
        f = lambda v: "/".join(f"{x:.1f}" for x in v)
        print(f"  {mode:14} {f(r['train_mape']):>22} {f(r['test_mape']):>22}")
        print(f"  {'OpDS (paper)':14} {ref[mode][0]:>22} {ref[mode][1]:>22}")
        print(f"  {'':14} test <d_spr> {r['test_d_spr']:.3f} [OpDS 0.644]"
              f"   inside-spread {r['test_inside']*100:.1f}%")
        print(f"  {'':14} ARB: violated cells {r['cell_frac']*100:.2f}%  arb-free surfaces "
              f"{r['arb_free']*100:.1f}%  mean depth {r['mean_depth']:.4f}  worst {r['worst_cell']:.3f}\n")

    dst = ROOT / "checkpoints" / run_name / "eval_opds_benchmark.json"
    json.dump(out, open(dst, "w"), indent=2)
    print(f"saved -> {dst}")


if __name__ == "__main__":
    run(*sys.argv[1:])
