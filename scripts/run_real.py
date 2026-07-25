import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.run_finetuning import load_arb_grid, load_finetuned
from src.data_generation.grid import z_to_k
from src.model.finetune import finetune
from src.model.quote_loss import quote_arb_loss
from src.model.SSVI import fit_ssvi, predict_ssvi
from src.evaluation.surface_eval import eval_arbitrage_fine, eval_real_surfaces
from src.real_data.dataloader import build_task, make_real_eval_set, temporal_split

ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = ROOT / "datasets" / "val"
EVAL_DIR = ROOT / "datasets" / "eval"
VAL_SEED = 0
EVAL_SEED = 1
N_HELDOUT = 1700  # ~arb-grid-sized NLL query; thinnest filtered day has 2439 quotes, so this still leaves room for n_ctx
VAL_CTX_SIZES = [5, 10, 20, 40, 60]
EVAL_CTX_SIZES = [3, 5, 10, 20, 40, 60]


def load_real_val(pool, cfg, rebuild=False, selffit=False):
    path = VAL_DIR / ("real_selffit.pkl" if selffit else "real.pkl")
    if path.exists() and not rebuild:
        return pickle.load(open(path, "rb"))
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(VAL_SEED)
    val = make_real_eval_set(pool, VAL_CTX_SIZES, cfg=cfg, n_heldout=N_HELDOUT, selffit=selffit)
    pickle.dump(val, open(path, "wb"))
    return val


def load_real_eval(pool, pool_name, rebuild=False):
    # cfg=None: query = held-out quotes only (no arb grid); the arb pass uses the frozen
    # synthetic arb grid separately, same as the synthetic eval stage
    path = EVAL_DIR / f"real_{pool_name}.pkl"
    if path.exists() and not rebuild:
        return pickle.load(open(path, "rb"))
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    eval_set = {}
    for n_ctx in EVAL_CTX_SIZES:
        np.random.seed(EVAL_SEED * 1000 + n_ctx)
        eval_set[n_ctx] = make_real_eval_set(pool, [n_ctx], cfg=None, n_heldout=N_HELDOUT)
    pickle.dump(eval_set, open(path, "wb"))
    return eval_set


def _real_baseline_one(args):
    (X_tr, y_tr), (X_held, y_held), cfg = args
    nc = len(y_tr) // 2
    bid = np.concatenate([y_tr[:nc], y_held[:, 0]])
    ask = np.concatenate([y_tr[nc:], y_held[:, 1]])
    mid = (bid + ask) / 2

    z_ctx, tau_ctx = X_tr[:nc, 0], X_tr[:nc, 1]
    s = (y_tr[nc:] - y_tr[:nc]) / 2
    w = 1 / np.maximum(2 * mid[:nc] * tau_ctx * s, 1e-10)
    X_k = np.column_stack([z_to_k(z_ctx, tau_ctx), tau_ctx])
    params, _ = fit_ssvi(X_k, mid[:nc], cfg, weights=w)

    z_all = np.concatenate([z_ctx, X_held[:, 0]])
    tau_all = np.concatenate([tau_ctx, X_held[:, 1]])
    pred = predict_ssvi(params, tau_all, z_to_k(z_all, tau_all))
    inside = (pred >= bid) & (pred <= ask)
    err = np.abs(pred - mid)
    return err[nc:].mean(), inside[nc:].mean(), err[:nc].mean(), inside[:nc].mean()


def load_real_baselines(eval_set, cfg, pool_name, rebuild=False, n_jobs=None):
    path = EVAL_DIR / f"real_baselines_{pool_name}.json"
    if path.exists() and not rebuild:
        return {int(n): v for n, v in json.load(open(path)).items()}
    jobs, slot_sizes = [], []
    for n_ctx, (tr, te) in eval_set.items():
        jobs += [(t, h, cfg) for t, h in zip(tr, te)]
        slot_sizes.append((n_ctx, len(tr)))
    with ProcessPoolExecutor(max_workers=n_jobs or os.cpu_count()) as ex:
        results = list(ex.map(_real_baseline_one, jobs, chunksize=4))
    out, pos = {}, 0
    for n_ctx, n in slot_sizes:
        cols = list(zip(*results[pos:pos + n]))
        pos += n
        out[n_ctx] = dict(zip(("mae_held", "inside_held", "mae_ctx", "inside_ctx"),
                              (float(np.mean(c)) for c in cols)))
    json.dump(out, open(path, "w"), indent=2)
    return out


def _write_eval_txt(path, run_name, which, pool_name, results, arb_results, baselines):
    lines = [f"{run_name} ({which}.pt), real data, pool={pool_name}",
             "\n=== held-out quotes ===",
             f"{'n_ctx':>6} {'FT MAE':>8} {'SSVI MAE':>9} {'FT ins%':>8} {'SSVI ins%':>10}"]
    for n_ctx, r in results.items():
        b = baselines[n_ctx]
        lines.append(f"{n_ctx:>6} {r['mae_held']:>8.4f} {b['mae_held']:>9.4f}"
                     f" {r['inside_held']*100:>7.1f}% {b['inside_held']*100:>9.1f}%")
    lines += ["\n=== context quotes ===",
              f"{'n_ctx':>6} {'FT MAE':>8} {'SSVI MAE':>9} {'FT ins%':>8} {'SSVI ins%':>10}"]
    for n_ctx, r in results.items():
        b = baselines[n_ctx]
        lines.append(f"{n_ctx:>6} {r['mae_ctx']:>8.4f} {b['mae_ctx']:>9.4f}"
                     f" {r['inside_ctx']*100:>7.1f}% {b['inside_ctx']*100:>9.1f}%")

    lines.append("\n\nArbitrage")
    lines.append(f"{'n_ctx':>6} {'cell_frac':>9} {'mean_depth':>11} {'worst_cell':>11} {'arb_free%':>10}")
    for n_ctx, r in arb_results.items():
        lines.append(f"{n_ctx:>6} {r['cell_frac']*100:>8.2f}% {r['mean_depth']:>11.4f}"
                     f" {r['worst_cell']:>11.4f} {r['arb_free']*100:>9.1f}%")
    open(path, "w").write("\n".join(lines) + "\n")


def run_eval(run_name, pool, pool_name, cfg, device, which="final", rebuild_eval=False, baseline_jobs=None,
             y_mean=0.300, y_scale=0.135, feature_scale=None):
    eval_set = load_real_eval(pool, pool_name, rebuild=rebuild_eval)
    arb_rows = load_arb_grid(cfg)
    model, state = load_finetuned(run_name, device, which=which)

    results, arb_results = {}, {}
    for i, (n_ctx, (tr, te)) in enumerate(eval_set.items(), 1):
        print(f"[{i}/{len(eval_set)}] n_ctx={n_ctx} ...", flush=True)
        results[n_ctx] = eval_real_surfaces(model, tr, te, reload_state=state, iv_max=cfg["iv_max"],
                                             y_mean=y_mean, y_scale=y_scale, feature_scale=feature_scale)
        cell_frac, mean_depth, worst_cell, arb_free = eval_arbitrage_fine(
            model, tr, cfg, arb_rows, reload_state=state, iv_max=cfg["iv_max"],
            y_mean=y_mean, y_scale=y_scale, feature_scale=feature_scale)
        arb_results[n_ctx] = dict(cell_frac=float(cell_frac), mean_depth=float(mean_depth),
                                  worst_cell=float(worst_cell), arb_free=float(arb_free))

    baselines = load_real_baselines(eval_set, cfg, pool_name, rebuild=rebuild_eval, n_jobs=baseline_jobs)

    out_dir = ROOT / "checkpoints" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(dict(real=results, arb=arb_results, baselines=baselines),
              open(out_dir / f"eval_real_{pool_name}.json", "w"), indent=2)
    txt = out_dir / f"eval_real_{pool_name}.txt"
    _write_eval_txt(txt, run_name, which, pool_name, results, arb_results, baselines)
    print(f"\nsaved eval -> {txt}")
    print(open(txt).read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--init-from", default=None,
                   help="synthetic run name to warm-start from (checkpoints/<name>/final.pt); "
                        "with --eval-only this evaluates that checkpoint zero-shot")
    p.add_argument("--which", default="final", choices=["final", "best"])
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--pool", default="val", choices=["val", "test"])
    p.add_argument("--start", default="2020-01-01")
    # parquet currently covers 2020-2023 contiguously plus a few stray 2026 days; default end
    # keeps the split inside the contiguous block
    p.add_argument("--end", default="2023-12-31")
    p.add_argument("--val-months", type=int, default=3)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--n-context", type=int, nargs=2, default=(3, 60), metavar=("LO", "HI"))
    # gs4_15k benchmark recipe: 500 epochs x 480 surfaces, batch 16, group 4 -> 15k steps
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--n-surfaces", type=int, default=480)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rebuild-val", action="store_true")
    p.add_argument("--rebuild-eval", action="store_true")
    p.add_argument("--baseline-jobs", type=int, default=None)
    p.add_argument("--wandb-project", default="volpfn", help="W&B project name (empty string to disable)")
    p.add_argument("--wandb-entity", default="volpfn")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--selffit", action="store_true",
                    help="OpDS-style: no held-out split, train/val on exact-match self-consistency "
                         "at context points instead of held-out generalization (eval is unaffected, "
                         "still genuine held-out)")
    p.add_argument("--y-source", default="synthetic", choices=["synthetic", "real"],
                    help="which cfg y_mean/y_scale (and z_mean/tau_mean if --feature-zscore) to use "
                         "for the global z-score - real SPXW quotes are narrower-scaled than the "
                         "synthetic prior (see config.yaml)")
    p.add_argument("--feature-zscore", action="store_true",
                    help="also z-score z/tau input features globally, using the same --y-source")
    p.add_argument("--global-squashing", action="store_true",
                    help="robust median/IQR global rescale for y AND z/tau (same --y-source) "
                         "instead of mean/std - no clip (z/tau are already domain-bounded)")
    args = p.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    suffix = "_real" if args.y_source == "real" else ""
    feature_scale = None
    if args.global_squashing:
        y_mean, y_scale = cfg["y_median" + suffix], cfg["y_iqr" + suffix]
        feature_scale = (cfg["z_median" + suffix], cfg["z_iqr" + suffix],
                          cfg["tau_median" + suffix], cfg["tau_iqr" + suffix])
    else:
        y_mean, y_scale = cfg["y_mean" + suffix], cfg["y_scale" + suffix]
        if args.feature_zscore:
            feature_scale = (cfg["z_mean" + suffix], cfg["z_scale" + suffix],
                              cfg["tau_mean" + suffix], cfg["tau_scale" + suffix])
    train_pool, val_pool, test_pool = temporal_split(
        args.start, args.end, val_months=args.val_months, test_months=args.test_months, cfg=cfg)
    print(f"pool sizes: train={len(train_pool)} val={len(val_pool)} test={len(test_pool)}")

    init_state = None
    if args.init_from is not None:
        init_state = torch.load(ROOT / "checkpoints" / args.init_from / "final.pt", map_location="cpu")

    if not args.eval_only:
        val_data = load_real_val(val_pool, cfg, rebuild=args.rebuild_val, selffit=args.selffit)
        data_provider = partial(build_task, train_pool, n_context=tuple(args.n_context),
                                cfg=cfg, n_heldout=N_HELDOUT, size_group=args.group_size, selffit=args.selffit)
        loss_fn = partial(quote_arb_loss, cfg=cfg, lambda_cal=10.0, lambda_bf=10.0,
                          lambda_reg_z=0.01, lambda_reg_r=0.01, return_parts=True)
        finetune(
            data_provider, run_name=args.run_name, n_epochs=args.epochs,
            n_surfaces_per_epoch=args.n_surfaces, batch_size=args.batch_size,
            group_size=args.group_size, val_group_size=args.group_size,
            val_data=val_data, val_every=args.val_every, loss_fn=loss_fn, device=args.device,
            iv_max=cfg["iv_max"], y_mean=y_mean, y_scale=y_scale, feature_scale=feature_scale,
            wandb_project=args.wandb_project or None, wandb_entity=args.wandb_entity,
            init_state=init_state, lr=args.lr,
        )
        eval_run_name = args.run_name
    else:
        eval_run_name = args.init_from or args.run_name

    pool = val_pool if args.pool == "val" else test_pool
    run_eval(eval_run_name, pool, args.pool, cfg, args.device,
             which=args.which, rebuild_eval=args.rebuild_eval, baseline_jobs=args.baseline_jobs,
             y_mean=y_mean, y_scale=y_scale, feature_scale=feature_scale)


if __name__ == "__main__":
    main()
