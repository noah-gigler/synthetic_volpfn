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
from tabpfn import TabPFNRegressor

from src.data_generation.grid import Grid, sample_arb_grid, z_to_k
from src.data_generation.noise import (
    make_noisy_stratified_eval_set, make_quote_eval_set,
    noisy_data_preparation, quote_data_preparation,
)
from src.model.finetune import finetune
from src.model.quote_loss import quote_arb_loss
from src.model.SSVI import fit_ssvi, predict_ssvi
from src.evaluation.surface_eval import eval_arbitrage_fine, eval_surfaces

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"
VAL_DIR = DATA_DIR / "val"
EVAL_DIR = DATA_DIR / "eval"
VAL_SEED = 0
EVAL_SEED = 1
N_HELDOUT = 15
EVAL_N = 512
VAL_CTX_SIZES = [5, 10, 20, 40, 60]
EVAL_CTX_SIZES = [3, 5, 10, 20, 40, 60]

EVAL_REGIMES = [0, 0.5, 1, 2]


def _split_by_regime(build, n_total):
    # n_total surfaces split into equal chunks across EVAL_REGIMES, so val is stratified
    # by noise regime instead of a single random draw
    n = n_total // len(EVAL_REGIMES)
    train, test = [], []
    for m in EVAL_REGIMES:
        tr, te = build(n, m)
        train += tr
        test += te
    return train, test


def _supervised_val(cfg):
    return _split_by_regime(lambda n, m: make_noisy_stratified_eval_set(cfg, n, VAL_CTX_SIZES, regime=m), 128)


def _arb_val(cfg):
    def build(n, m):
        sets = [make_quote_eval_set(cfg, n, s, N_HELDOUT, regime=m, size_group=GROUP_SIZE) for s in VAL_CTX_SIZES]
        return sum((s[0] for s in sets), []), sum((s[1] for s in sets), [])
    return _split_by_regime(build, 128)


GROUP_SIZE = 8
BATCH_SIZE = 32
EXPERIMENTS = {
    "supervised": dict(
        provider=lambda cfg, n_ctx: partial(noisy_data_preparation, cfg, n_context=n_ctx,  size_group=GROUP_SIZE),
        val=_supervised_val,
        loss=lambda cfg: None,
        group_size=GROUP_SIZE, batch_size=BATCH_SIZE, val_group_size=128,
    ),
    "arb": dict(
        provider=lambda cfg, n_ctx: partial(
            quote_data_preparation, cfg, n_context=n_ctx, n_heldout=N_HELDOUT, size_group=GROUP_SIZE),
        val=_arb_val,
        loss=lambda cfg: partial(quote_arb_loss, cfg=cfg, lambda_cal=1.0, lambda_bf=1.0),
        group_size=GROUP_SIZE, batch_size=BATCH_SIZE, val_group_size=GROUP_SIZE,
    ),
}


def load_eval(cfg, rebuild=False):
    # frozen eval set, shared across every experiment (both train on the noisy/bid-ask
    # schema); surfaces are seeded per n_ctx only, so it's reused as-is across models.
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / "noisy.pkl"
    if path.exists() and not rebuild:
        return pickle.load(open(path, "rb"))
    eval_set = {}
    for m in EVAL_REGIMES:
        eval_set[m] = {}
        for n_ctx in EVAL_CTX_SIZES:
            np.random.seed(EVAL_SEED * 1000 + n_ctx)  # per-n_ctx -> same truth across regimes
            eval_set[m][n_ctx] = noisy_data_preparation(cfg, EVAL_N, n_ctx, regime=m)
    pickle.dump(eval_set, open(path, "wb"))
    return eval_set


def load_arb_grid(cfg, rebuild=False):
    # frozen once (same grid reused across every checkpoint's arb eval), unlike training's
    # per-surface random grid -> every surface shares the same query shape and can be batched
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / "arb_grid.pkl"
    if path.exists() and not rebuild:
        return pickle.load(open(path, "rb"))
    np.random.seed(EVAL_SEED)
    rows = sample_arb_grid(cfg, jitter=True, r_b_step_range=(0.019, 0.025))
    pickle.dump(rows, open(path, "wb"))
    return rows


def load_val(experiment, cfg, rebuild=False):
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    path = VAL_DIR / f"{experiment}.pkl"
    if path.exists() and not rebuild:
        return pickle.load(open(path, "rb"))
    np.random.seed(VAL_SEED)
    val = EXPERIMENTS[experiment]["val"](cfg)
    pickle.dump(val, open(path, "wb"))
    return val


def load_finetuned(run_name, device, which="final"):
    model = TabPFNRegressor(
        fit_mode="fit_preprocessors", n_estimators=1,
        inference_config={"FINGERPRINT_FEATURE": False},
    )
    model._initialize_model_variables()
    state = torch.load(ROOT / "checkpoints" / run_name / f"{which}.pt", map_location=device)
    model.model_.load_state_dict(state)
    return model, state


def _split_quotes(train):
    # (X[:, :2], mid, half_spread) per surface from the stacked bid/ask context rows
    out = []
    for X, y in train:
        n = len(y) // 2
        out.append((X[:n, :2], (y[:n] + y[n:]) / 2, (y[n:] - y[:n]) / 2))
    return out


def _baseline_one(args):
    # one surface's SSVI refit; top-level (not a closure) so it's picklable for ProcessPoolExecutor
    (X2, mq, s), (Xq, yq), cfg_dict = args
    g = Grid(cfg_dict)
    w = 1 / np.maximum(2 * mq * X2[:, 1] * s, 1e-10)
    # feature col 0 is z; SSVI fits/predicts in physical strike k
    X2_k = np.column_stack([z_to_k(X2[:, 0], X2[:, 1]), X2[:, 1]])
    params, _ = fit_ssvi(X2_k, mq, cfg_dict, weights=w)
    pred = predict_ssvi(params, g.ttms[:, None], g.k.reshape(g.shape)).ravel()
    wls = np.mean(np.abs(pred - yq))
    wls_mape = np.mean(np.abs((yq - pred) / yq)) * 100
    idx = [np.where((Xq[:, 0] == X2[i, 0]) & (Xq[:, 1] == X2[i, 1]))[0][0] for i in range(len(mq))]
    mid = np.abs(mq - yq[idx]).mean()
    return wls, wls_mape, mid


def _baselines(eval_set, cfg, n_jobs=None):
    # model-independent MAE baselines on the noisy eval set (needs bid/ask quotes); the SSVI
    # refit is CPU-only and independent per surface, so it's dispatched across processes -
    # doesn't touch the GPU at all, can run on a separate CPU-heavy allocation from the eval passes
    n_jobs = n_jobs or os.cpu_count()
    jobs = []
    slot_sizes = []
    for m, by_ctx in eval_set.items():
        for n_ctx, (tr, te) in by_ctx.items():
            quotes = _split_quotes(tr)
            jobs += [(q, t, cfg) for q, t in zip(quotes, te)]
            slot_sizes.append((m, n_ctx, len(quotes)))

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        results = list(ex.map(_baseline_one, jobs, chunksize=4))

    out = {}
    pos = 0
    for m, n_ctx, n in slot_sizes:
        wls, wls_mape, mid = zip(*results[pos:pos + n])
        pos += n
        out.setdefault(m, {})[n_ctx] = dict(
            refit_wls=float(np.mean(wls)), refit_wls_mape=float(np.mean(wls_mape)), mid=float(np.mean(mid)))
    return out


def load_baselines(cfg, eval_set, rebuild=False, n_jobs=None):
    path = EVAL_DIR / "noisy_baselines.json"
    if path.exists() and not rebuild:
        raw = json.load(open(path))
        return {float(m): {int(n): v for n, v in rows.items()} for m, rows in raw.items()}
    base = _baselines(eval_set, cfg, n_jobs=n_jobs)
    json.dump(base, open(path, "w"), indent=2, default=str)
    return base


def _write_eval_txt(path, run_name, which, rho, results, arb_results, baselines):
    lines = [f"{run_name} ({which}.pt), rho={rho}"]

    for m, rows in results.items():
        lines.append(f"\n=== regime m={m} ===")
        lines.append(f"{'n_ctx':>6} {'FT MAE':>8} {'SSVI MAE':>9} {'FT MAPE%':>9} {'SSVI MAPE%':>11}")
        for n_ctx, r in rows.items():
            b = baselines[float(m)][int(n_ctx)]
            lines.append(f"{n_ctx:>6} {r['mae']:>8.4f} {b['refit_wls']:>9.4f}"
                         f" {r['mape']:>9.2f} {b['refit_wls_mape']:>11.2f}")

    lines.append("\n\nArbitrage")
    for m, rows in arb_results.items():
        lines.append(f"\n=== regime m={m} ===")
        lines.append(f"{'n_ctx':>6} {'cell_frac':>9} {'mean_depth':>11} {'worst_cell':>11} {'arb_free%':>10}")
        for n_ctx, r in rows.items():
            lines.append(f"{n_ctx:>6} {r['cell_frac']*100:>8.2f}% {r['mean_depth']:>11.4f}"
                         f" {r['worst_cell']:>11.4f} {r['arb_free']*100:>9.1f}%")

    open(path, "w").write("\n".join(lines) + "\n")


def run_eval(run_name, cfg, device, rebuild_eval=False, which="final", rho=0.0, baseline_jobs=None):
    eval_set = load_eval(cfg, rebuild=rebuild_eval)
    arb_rows = load_arb_grid(cfg, rebuild=rebuild_eval)
    model, state = load_finetuned(run_name, device, which=which)
    results, arb_results = {}, {}
    slots = [(m, n_ctx) for m, by_ctx in eval_set.items() for n_ctx in by_ctx]
    for i, (m, n_ctx) in enumerate(slots, 1):
        tr, te = eval_set[m][n_ctx]
        print(f"[{i}/{len(slots)}] regime={m} n_ctx={n_ctx} ...", flush=True)
        mae, mape, cal, bf = eval_surfaces(model, tr, te, cfg, reload_state=state)
        results.setdefault(str(m), {})[n_ctx] = dict(
            mae=float(mae), mape=float(mape), cal_viol=float(cal), bf_viol=float(bf))

        cell_frac, mean_depth, worst_cell, arb_free = eval_arbitrage_fine(
            model, tr, cfg, arb_rows, reload_state=state)
        arb_results.setdefault(str(m), {})[n_ctx] = dict(
            cell_frac=float(cell_frac), mean_depth=float(mean_depth),
            worst_cell=float(worst_cell), arb_free=float(arb_free))

    baselines = load_baselines(cfg, eval_set, rebuild=rebuild_eval, n_jobs=baseline_jobs)

    out_dir = ROOT / "checkpoints" / run_name
    json.dump(dict(mae=results, arb=arb_results), open(out_dir / "eval.json", "w"), indent=2)
    _write_eval_txt(out_dir / "eval.txt", run_name, which, rho, results, arb_results, baselines)
    print(f"\nsaved eval -> {out_dir/'eval.json'}  and  {out_dir/'eval.txt'}")
    print(open(out_dir / "eval.txt").read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=list(EXPERIMENTS))
    p.add_argument("--run-name", default=None)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n-surfaces", type=int, default=200)
    p.add_argument("--n-context", type=int, nargs=2, default=(3, 60), metavar=("LO", "HI"))
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--group-size", type=int, default=None)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rebuild-val", action="store_true")
    p.add_argument("--rebuild-eval", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--baseline-jobs", type=int, default=None,
                    help="CPU workers for the SSVI baseline refit (default: os.cpu_count())")
    p.add_argument("--wandb-project", default="volpfn", help="W&B project name (empty string to disable)")
    p.add_argument("--wandb-entity", default="volpfn", help="W&B team/entity name")
    args = p.parse_args()

    spec = EXPERIMENTS[args.experiment]
    run_name = args.run_name or f"{args.experiment}_{args.n_context[0]}_{args.n_context[1]}"
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))

    if not args.eval_only:
        val_data = load_val(args.experiment, cfg, rebuild=args.rebuild_val)
        data_provider = spec["provider"](cfg, tuple(args.n_context))
        loss_fn = spec["loss"](cfg)
        finetune(
            data_provider, run_name=run_name, n_epochs=args.epochs,
            n_surfaces_per_epoch=args.n_surfaces, batch_size=args.batch_size or spec["batch_size"],
            group_size=args.group_size or spec["group_size"], val_group_size=spec["val_group_size"],
            val_data=val_data, val_every=args.val_every, loss_fn=loss_fn, device=args.device,
            wandb_project=args.wandb_project or None, wandb_entity=args.wandb_entity,
        )

    run_eval(run_name, cfg, args.device, rebuild_eval=args.rebuild_eval, baseline_jobs=args.baseline_jobs)


if __name__ == "__main__":
    main()
