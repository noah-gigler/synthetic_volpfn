"""Evaluate a checkpoint on pure-Heston surfaces, with the Heston refit as the baseline.

Mirrors run_finetuning.run_eval's regime x n_ctx breakdown, but the eval set is drawn with
mixture.heston_frac=1.0 and the refit column is fit_heston instead of fit_ssvi - a model trained
on Heston data scored against an SSVI refit is measuring family mismatch, not fit quality. The
SSVI refit is kept as a third column so the two refits are directly comparable on the same
surfaces (the cross-family comparison that motivated this script).
"""
import argparse
import copy
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import yaml

from src.data_generation.grid import Grid, z_to_k
from src.data_generation.noise import noisy_data_preparation
from src.model.SSVI import fit_ssvi, predict_ssvi
from src.model.heston import fit_heston, predict_heston
from src.evaluation.surface_eval import eval_arbitrage_fine, eval_surfaces, eval_uncertainty
from scripts.run_finetuning import (
    EVAL_DIR, EVAL_SEED, ROOT, load_arb_grid, load_finetuned, _split_quotes,
)

REFITS = {
    "hes": dict(lambda_prior=0.0, anchor_v0=False),
    "hesmap": dict(lambda_prior=1.0, anchor_v0=False),
}


def heston_cfg(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("mixture", {})["heston_frac"] = 1.0
    return cfg


def load_heston_eval(cfg, regimes, ctx_sizes, eval_n, rebuild=False, tag=""):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"heston{tag}.pkl"
    if path.exists() and not rebuild:
        return pickle.load(open(path, "rb"))
    hcfg = heston_cfg(cfg)
    eval_set = {}
    for m in regimes:
        eval_set[m] = {}
        for n_ctx in ctx_sizes:
            np.random.seed(EVAL_SEED * 1000 + n_ctx)  # per-n_ctx -> same truth across regimes
            eval_set[m][n_ctx] = noisy_data_preparation(hcfg, eval_n, n_ctx, regime=m)
    pickle.dump(eval_set, open(path, "wb"))
    return eval_set


def _baseline_one(args):
    (X2, mq, s), (Xq, yq), cfg_dict = args
    g = Grid(cfg_dict)
    w = 1 / np.maximum(2 * mq * X2[:, 1] * s, 1e-10)
    X2_k = np.column_stack([z_to_k(X2[:, 0], X2[:, 1]), X2[:, 1]])
    # Heston surfaces carry NaN in the deep-OTM/shortest-maturity corner (float64 price
    # underflow, see data_preperation.generate_surfaces) - those cells are excluded from the
    # refit MAE exactly as eval_surfaces excludes them for the model
    valid = np.isfinite(yq)

    out = {}
    ps, _ = fit_ssvi(X2_k, mq, cfg_dict, weights=w)
    pred = predict_ssvi(ps, g.ttms[:, None], g.k.reshape(g.shape)).ravel()
    m = valid & np.isfinite(pred)
    out["ssvi"] = float(np.mean(np.abs(pred[m] - yq[m])))
    out["ssvi_mape"] = float(np.mean(np.abs((yq[m] - pred[m]) / yq[m])) * 100)

    for tag, kw in REFITS.items():
        ph, _ = fit_heston(X2_k, mq, cfg_dict, weights=w, **kw)
        pred = predict_heston(ph, g.tau, g.k)
        m = valid & np.isfinite(pred)
        out[tag] = float(np.mean(np.abs(pred[m] - yq[m])))
        out[tag + "_mape"] = float(np.mean(np.abs((yq[m] - pred[m]) / yq[m])) * 100)

    idx = [np.where((Xq[:, 0] == X2[i, 0]) & (Xq[:, 1] == X2[i, 1]))[0][0] for i in range(len(mq))]
    out["mid"] = float(np.abs(mq - yq[idx]).mean())
    return out


def load_heston_baselines(cfg, eval_set, rebuild=False, n_jobs=None, tag=""):
    path = EVAL_DIR / f"heston_baselines{tag}.json"
    if path.exists() and not rebuild:
        raw = json.load(open(path))
        return {float(m): {int(n): v for n, v in rows.items()} for m, rows in raw.items()}

    hcfg = heston_cfg(cfg)
    jobs, slots = [], []
    for m, by_ctx in eval_set.items():
        for n_ctx, (tr, te) in by_ctx.items():
            quotes = _split_quotes(tr)
            jobs += [(q, t, hcfg) for q, t in zip(quotes, te)]
            slots.append((m, n_ctx, len(quotes)))

    n_jobs = n_jobs or os.cpu_count()
    print(f"{len(jobs)} refits x {1 + len(REFITS)} models on {n_jobs} workers", flush=True)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        results = list(ex.map(_baseline_one, jobs, chunksize=4))

    out, pos = {}, 0
    for m, n_ctx, n in slots:
        chunk = results[pos:pos + n]
        pos += n
        # float keys either way, so the cached (json-reloaded) and fresh paths index the same
        out.setdefault(float(m), {})[int(n_ctx)] = {
            k: float(np.mean([c[k] for c in chunk])) for k in chunk[0]}
    json.dump(out, open(path, "w"), indent=2, default=str)
    return out


def _write_txt(path, run_name, which, results, arb_results, uq_results, baselines):
    lines = [f"{run_name} ({which}.pt) - PURE HESTON eval set, Heston + SSVI refit baselines"]

    # driven by `baselines`, not `results`, so --skip-model still prints the refit comparison
    for m in sorted(baselines):
        lines.append(f"\n=== regime m={m:g} ===")
        lines.append(f"{'n_ctx':>6} {'FT MAE':>8} {'Hes MAE':>9} {'HesMAP':>9} {'SSVI MAE':>9}"
                     f" {'FT MAPE%':>9} {'Hes MAPE%':>10} {'SSVI MAPE%':>11}")
        for n_ctx in sorted(baselines[m]):
            b = baselines[m][n_ctx]
            r = results.get(f"{m:g}", {}).get(n_ctx) or {}
            ft_mae = f"{r['mae']:>8.4f}" if r else f"{'-':>8}"
            ft_mape = f"{r['mape']:>9.2f}" if r else f"{'-':>9}"
            lines.append(f"{n_ctx:>6} {ft_mae} {b['hes']:>9.4f} {b['hesmap']:>9.4f}"
                         f" {b['ssvi']:>9.4f} {ft_mape} {b['hes_mape']:>10.2f}"
                         f" {b['ssvi_mape']:>11.2f}")

    lines.append("\n\nArbitrage")
    for m, rows in arb_results.items():
        lines.append(f"\n=== regime m={m} ===")
        lines.append(f"{'n_ctx':>6} {'cell_frac':>9} {'mean_depth':>11} {'worst_cell':>11} {'arb_free%':>10}")
        for n_ctx, r in rows.items():
            lines.append(f"{n_ctx:>6} {r['cell_frac']*100:>8.2f}% {r['mean_depth']:>11.4f}"
                         f" {r['worst_cell']:>11.4f} {r['arb_free']*100:>9.1f}%")

    lines.append("\n\nUncertainty (predictive-distribution calibration, see eval_uncertainty)")
    for m, rows in uq_results.items():
        lines.append(f"\n=== regime m={m} ===")
        lines.append(f"{'n_ctx':>6} {'CRPS':>8} {'pinball05':>10} {'pinball95':>10} {'width90%':>9}")
        for n_ctx, r in rows.items():
            lines.append(f"{n_ctx:>6} {r['crps']:>8.4f} {r['pinball'][0.05]:>10.4f}"
                         f" {r['pinball'][0.95]:>10.4f} {r['mean_interval_width']:>9.4f}")

    open(path, "w").write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_name")
    p.add_argument("--which", default="final", choices=["final", "best"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--regimes", type=float, nargs="+", default=[0, 0.5, 1, 2])
    p.add_argument("--ctx-sizes", type=int, nargs="+", default=[3, 5, 10, 20, 40, 60])
    p.add_argument("--baseline-jobs", type=int, default=None)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--tag", default="", help="eval-set cache suffix; use for smoke runs so the "
                                             "full frozen set isn't overwritten")
    p.add_argument("--skip-model", action="store_true", help="baselines only (CPU, no GPU needed)")
    p.add_argument("--y-mean", type=float, default=0.300)
    p.add_argument("--y-scale", type=float, default=0.135)
    a = p.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    regimes = [int(m) if float(m).is_integer() else m for m in a.regimes]
    eval_set = load_heston_eval(cfg, regimes, a.ctx_sizes, a.eval_n, rebuild=a.rebuild, tag=a.tag)

    results, arb_results, uq_results, pit_arrays = {}, {}, {}, {}
    if not a.skip_model:
        arb_rows = load_arb_grid(cfg)
        model, state = load_finetuned(a.run_name, a.device, which=a.which)
        slots = [(m, n) for m, by_ctx in eval_set.items() for n in by_ctx]
        for i, (m, n_ctx) in enumerate(slots, 1):
            tr, te = eval_set[m][n_ctx]
            print(f"[{i}/{len(slots)}] regime={m} n_ctx={n_ctx} ...", flush=True)
            mae, mape, cal, bf = eval_surfaces(model, tr, te, cfg, reload_state=state,
                                               iv_max=cfg["iv_max"], y_mean=a.y_mean, y_scale=a.y_scale)
            results.setdefault(str(m), {})[n_ctx] = dict(
                mae=float(mae), mape=float(mape), cal_viol=float(cal), bf_viol=float(bf))

            cell_frac, mean_depth, worst_cell, arb_free = eval_arbitrage_fine(
                model, tr, cfg, arb_rows, reload_state=state, iv_max=cfg["iv_max"],
                y_mean=a.y_mean, y_scale=a.y_scale)
            arb_results.setdefault(str(m), {})[n_ctx] = dict(
                cell_frac=float(cell_frac), mean_depth=float(mean_depth),
                worst_cell=float(worst_cell), arb_free=float(arb_free))

            uq = eval_uncertainty(model, tr, te, reload_state=state, iv_max=cfg["iv_max"],
                                  y_mean=a.y_mean, y_scale=a.y_scale)
            uq_results.setdefault(str(m), {})[n_ctx] = dict(
                crps=uq["crps"], pinball=uq["pinball"], mean_interval_width=uq["mean_interval_width"])
            pit_arrays[f"{m}_{n_ctx}"] = uq["pit"]

    baselines = load_heston_baselines(cfg, eval_set, rebuild=a.rebuild,
                                      n_jobs=a.baseline_jobs, tag=a.tag)

    out_dir = ROOT / "checkpoints" / a.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(dict(mae=results, arb=arb_results, uq=uq_results, baselines=baselines),
              open(out_dir / f"eval_heston{a.tag}.json", "w"), indent=2, default=str)
    if pit_arrays:
        np.savez(out_dir / f"pit_heston{a.tag}.npz", **pit_arrays)
    _write_txt(out_dir / f"eval_heston{a.tag}.txt", a.run_name, a.which,
               results, arb_results, uq_results, baselines)
    print(open(out_dir / f"eval_heston{a.tag}.txt").read())


if __name__ == "__main__":
    main()
