import argparse
import json
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from tabpfn import TabPFNRegressor

from src.data_generation.data_preperation import (
    data_preparation, make_stratified_eval_set,
)
from src.data_generation.grid import Grid, z_to_k
from src.data_generation.noise import (
    make_noisy_stratified_eval_set, make_quote_eval_set,
    noisy_data_preparation, quote_data_preparation,
)
from src.model.finetune import finetune
from src.model.quote_loss import quote_arb_loss
from src.model.SSVI import fit_ssvi, predict_ssvi
from src.evaluation.surface_eval import eval_surfaces

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"
VAL_DIR = DATA_DIR / "val"
EVAL_DIR = DATA_DIR / "eval"
VAL_SEED = 0
EVAL_SEED = 1
N_HELDOUT = 15
EVAL_N = 50
VAL_CTX_SIZES = [5, 10, 20, 40, 60]
EVAL_CTX_SIZES = [3, 5, 10, 20, 40, 60]

# eval sets are frozen per schema and shared across every model with that schema, so
# their tables are row-aligned and combinable. surfaces are seeded per n_ctx only, so
# clean and noisy schemas share the same underlying truth (differing only in context).
EVAL_SCHEMAS = {
    "clean": dict(
        provider=lambda cfg, n, n_ctx, m: data_preparation(cfg, n, n_ctx),
        regimes=[None],
    ),
    "noisy": dict(
        provider=lambda cfg, n, n_ctx, m: noisy_data_preparation(cfg, n, n_ctx, regime=m),
        regimes=[0.5, 1, 2],
    ),
}


def _clean_val(cfg):
    return make_stratified_eval_set(cfg, 20, VAL_CTX_SIZES)


def _supervised_val(cfg):
    return make_noisy_stratified_eval_set(cfg, 20, VAL_CTX_SIZES)


def _arb_val(cfg):
    sets = [make_quote_eval_set(cfg, 8, s, N_HELDOUT) for s in VAL_CTX_SIZES]
    return sum((s[0] for s in sets), []), sum((s[1] for s in sets), [])


EXPERIMENTS = {
    "clean": dict(
        provider=lambda cfg, n_ctx: partial(data_preparation, cfg, n_context=n_ctx),
        val=_clean_val,
        loss=lambda cfg, grid: None,
        eval_schema="clean",
        group_size=1, batch_size=4,
    ),
    "supervised": dict(
        provider=lambda cfg, n_ctx: partial(noisy_data_preparation, cfg, n_context=n_ctx),
        val=_supervised_val,
        loss=lambda cfg, grid: None,
        eval_schema="noisy",
        group_size=4, batch_size=8,
    ),
    "arb": dict(
        provider=lambda cfg, n_ctx: partial(
            quote_data_preparation, cfg, n_context=n_ctx, n_heldout=N_HELDOUT, size_group=4),
        val=_arb_val,
        loss=lambda cfg, grid: partial(quote_arb_loss, grid_shape=grid, lambda_cal=1.0, lambda_bf=1.0),
        eval_schema="noisy",
        group_size=4, batch_size=8,
    ),
}


def load_eval(schema, cfg, rebuild=False):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"{schema}.pkl"
    if path.exists() and not rebuild:
        return pickle.load(open(path, "rb"))
    spec = EVAL_SCHEMAS[schema]
    eval_set = {}
    for m in spec["regimes"]:
        eval_set[m] = {}
        for n_ctx in EVAL_CTX_SIZES:
            np.random.seed(EVAL_SEED * 1000 + n_ctx)  # per-n_ctx -> same truth across schemas/regimes
            eval_set[m][n_ctx] = spec["provider"](cfg, EVAL_N, n_ctx, m)
    pickle.dump(eval_set, open(path, "wb"))
    return eval_set


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


def _baselines(eval_set, cfg):
    # model-independent MAE baselines on the noisy eval set; needs bid/ask quotes so
    # returns None for the clean schema
    g = Grid(cfg)
    out = {}
    for m, by_ctx in eval_set.items():
        rows = {}
        for n_ctx, (tr, te) in by_ctx.items():
            if tr[0][0].shape[1] < 3:                  # clean context, no spread
                return None
            wls, mid = [], []
            for (X2, mq, s), (Xq, yq) in zip(_split_quotes(tr), te):
                w = 1 / np.maximum(2 * mq * X2[:, 1] * s, 1e-10)
                # feature col 0 is z; SSVI fits/predicts in physical strike k
                X2_k = np.column_stack([z_to_k(X2[:, 0], X2[:, 1]), X2[:, 1]])
                params, _ = fit_ssvi(X2_k, mq, cfg, weights=w)
                wls.append(np.mean(np.abs(predict_ssvi(params, g.ttms, g.k.reshape(g.shape)).ravel() - yq)))
                idx = [np.where((Xq[:, 0] == X2[i, 0]) & (Xq[:, 1] == X2[i, 1]))[0][0]
                       for i in range(len(mq))]
                mid.append(np.abs(mq - yq[idx]).mean())
            rows[n_ctx] = dict(refit_wls=float(np.mean(wls)), mid=float(np.mean(mid)))
        out[m] = rows
    return out


def load_baselines(schema, cfg, eval_set, rebuild=False):
    path = EVAL_DIR / f"{schema}_baselines.json"
    if path.exists() and not rebuild:
        raw = json.load(open(path))
        return {float(m): {int(n): v for n, v in rows.items()} for m, rows in raw.items()}
    base = _baselines(eval_set, cfg)
    if base is not None:
        json.dump(base, open(path, "w"), indent=2, default=str)
    return base


def _write_eval_txt(path, experiment, results, baselines):
    lines = []
    for m, rows in results.items():
        lines.append(f"\n=== regime={m} (MAE vs truth) ===")
        header = f"{'n_ctx':>6} {experiment:>12} {'refit WLS':>12} {'mid straw':>12}"
        lines.append(header if baselines is not None else f"{'n_ctx':>6} {experiment:>12}")
        for n_ctx, r in rows.items():
            if baselines is not None:
                b = baselines[float(m)][int(n_ctx)]
                lines.append(f"{n_ctx:>6} {r['mae']:>12.4f} {b['refit_wls']:>12.4f} {b['mid']:>12.4f}")
            else:
                lines.append(f"{n_ctx:>6} {r['mae']:>12.4f}")
    open(path, "w").write("\n".join(lines) + "\n")


def run_eval(experiment, run_name, cfg, device, rebuild_eval=False):
    spec = EXPERIMENTS[experiment]
    eval_set = load_eval(spec["eval_schema"], cfg, rebuild=rebuild_eval)
    model, state = load_finetuned(run_name, device)
    results = {}
    slots = [(m, n_ctx) for m, by_ctx in eval_set.items() for n_ctx in by_ctx]
    for i, (m, n_ctx) in enumerate(slots, 1):
        tr, te = eval_set[m][n_ctx]
        print(f"[{i}/{len(slots)}] regime={m} n_ctx={n_ctx} ...", flush=True)
        mae, mape, cal, bf = eval_surfaces(model, tr, te, cfg, reload_state=state)
        results.setdefault(str(m), {})[n_ctx] = dict(
            mae=float(mae), mape=float(mape), cal_viol=float(cal), bf_viol=float(bf))

    baselines = load_baselines(spec["eval_schema"], cfg, eval_set, rebuild=rebuild_eval)

    out_dir = ROOT / "checkpoints" / run_name
    json.dump(results, open(out_dir / "eval.json", "w"), indent=2)
    _write_eval_txt(out_dir / "eval.txt", experiment, results, baselines)
    print(f"\nsaved eval -> {out_dir/'eval.json'}  and  {out_dir/'eval.txt'}")
    print(open(out_dir / "eval.txt").read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=list(EXPERIMENTS))
    p.add_argument("--run-name", default=None)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n-surfaces", type=int, default=200)
    p.add_argument("--n-context", type=int, nargs=2, default=(3, 60), metavar=("LO", "HI"))
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rebuild-val", action="store_true")
    p.add_argument("--rebuild-eval", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    spec = EXPERIMENTS[args.experiment]
    run_name = args.run_name or f"{args.experiment}_{args.n_context[0]}_{args.n_context[1]}"
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))

    if not args.eval_only:
        val_data = load_val(args.experiment, cfg, rebuild=args.rebuild_val)
        data_provider = spec["provider"](cfg, tuple(args.n_context))
        loss_fn = spec["loss"](cfg, Grid(cfg).shape)
        finetune(
            data_provider, run_name=run_name, n_epochs=args.epochs,
            n_surfaces_per_epoch=args.n_surfaces, batch_size=spec["batch_size"],
            group_size=spec["group_size"], val_data=val_data, val_every=args.val_every,
            loss_fn=loss_fn, device=args.device,
        )

    run_eval(args.experiment, run_name, cfg, args.device, rebuild_eval=args.rebuild_eval)


if __name__ == "__main__":
    main()
