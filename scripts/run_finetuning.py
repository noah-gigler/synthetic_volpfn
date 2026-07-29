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
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn.constants import ModelVersion

from src.data_generation.grid import Grid, sample_arb_grid, z_to_k
from src.data_generation.noise import (
    make_noisy_stratified_eval_set, make_quote_eval_set,
    noisy_data_preparation, quote_data_preparation,
)
from src.model.finetune import finetune, crps_only_loss
from src.model.quote_loss import quote_arb_loss
from src.model.SSVI import fit_ssvi, predict_ssvi
from src.evaluation.surface_eval import eval_arbitrage_fine, eval_surfaces, eval_uncertainty

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"
VAL_DIR = DATA_DIR / "val"
EVAL_DIR = DATA_DIR / "eval"
VAL_SEED = 0
EVAL_SEED = 1
N_HELDOUT = 315
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


def _supervised_val(cfg, rho=0.0):
    return _split_by_regime(
        lambda n, m: make_noisy_stratified_eval_set(cfg, n, VAL_CTX_SIZES, regime=m, rho=rho), 128)


def _arb_val(cfg, rho=0.0):
    def build(n, m):
        sets = [make_quote_eval_set(cfg, n, s, N_HELDOUT, regime=m, size_group=GROUP_SIZE, rho=rho)
                for s in VAL_CTX_SIZES]
        return sum((s[0] for s in sets), []), sum((s[1] for s in sets), [])
    return _split_by_regime(build, 128)


GROUP_SIZE = 8
BATCH_SIZE = 32
EXPERIMENTS = {
    "supervised": dict(
        provider=lambda cfg, n_ctx, rho: partial(
            noisy_data_preparation, cfg, n_context=n_ctx, size_group=GROUP_SIZE, rho=rho),
        val=_supervised_val,
        loss=lambda cfg: None,
        group_size=GROUP_SIZE, batch_size=BATCH_SIZE, val_group_size=128,
    ),
    "arb": dict(
        provider=lambda cfg, n_ctx, rho: partial(
            quote_data_preparation, cfg, n_context=n_ctx, n_heldout=N_HELDOUT, size_group=GROUP_SIZE, rho=rho),
        val=_arb_val,
        loss=lambda cfg: partial(quote_arb_loss, cfg=cfg, lambda_cal=10.0, lambda_bf=10.0,
                                 lambda_reg_z=0.01, lambda_reg_r=0.01, return_parts=True),
        group_size=GROUP_SIZE, batch_size=BATCH_SIZE, val_group_size=GROUP_SIZE,
    ),
}


def _rho_tag(rho):
    # filename-safe cache suffix; a (lo, hi) range becomes e.g. rho0.0_1.0
    if isinstance(rho, (tuple, list)):
        return f"rho{rho[0]}_{rho[1]}"
    return f"rho{rho}"


def _legacy_tag(legacy):
    # legacy (the default) must not share caches with --calibrated-noise (or overwrite them on
    # --rebuild) - matches the pre-existing frozen "_oldnoise" files predating calibration
    return "_oldnoise" if legacy else ""


def load_eval(cfg, rebuild=False, rho=0.0, legacy=True):
    # frozen eval set, shared across every experiment (both train on the noisy/bid-ask
    # schema); surfaces are seeded per n_ctx only, so it's reused as-is across models.
    # rho=0 is the shared/default cache; any other rho gets its own file.
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stem = "noisy" if rho == 0.0 else f"noisy_{_rho_tag(rho)}"
    path = EVAL_DIR / f"{stem}{_legacy_tag(legacy)}.pkl"
    if path.exists() and not rebuild:
        return pickle.load(open(path, "rb"))
    eval_set = {}
    for m in EVAL_REGIMES:
        eval_set[m] = {}
        for n_ctx in EVAL_CTX_SIZES:
            np.random.seed(EVAL_SEED * 1000 + n_ctx)  # per-n_ctx -> same truth across regimes
            eval_set[m][n_ctx] = noisy_data_preparation(cfg, EVAL_N, n_ctx, regime=m, rho=rho)
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


def load_val(experiment, cfg, rebuild=False, rho=0.0, legacy=True):
    # rho=0 is the shared/frozen cache reused across every run of this experiment; any other
    # rho gets its own cache so it doesn't clobber that baseline
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    stem = experiment if rho == 0.0 else f"{experiment}_{_rho_tag(rho)}"
    path = VAL_DIR / f"{stem}{_legacy_tag(legacy)}.pkl"
    if path.exists() and not rebuild:
        return pickle.load(open(path, "rb"))
    np.random.seed(VAL_SEED)
    val = EXPERIMENTS[experiment]["val"](cfg, rho=rho)
    pickle.dump(val, open(path, "wb"))
    return val


def load_finetuned(run_name, device, which="final", model_version=None):
    common = dict(fit_mode="fit_preprocessors", n_estimators=1, categorical_features_indices=[], inference_config={
        "FINGERPRINT_FEATURE": False,
        "FEATURE_SHIFT_METHOD": None,
        "PREPROCESS_TRANSFORMS": [PreprocessorConfig("none", categorical_name="numeric")],
    })
    if model_version is not None:
        model = TabPFNRegressor.create_default_for_version(version=ModelVersion(model_version), **common)
    else:
        model = TabPFNRegressor(**common)
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
    # Heston surfaces carry NaN in the deep-OTM/short-tau corner (price underflow); averaging
    # over them makes the whole baseline NaN. SSVI surfaces have none, so this is a no-op there.
    m = np.isfinite(yq) & np.isfinite(pred)
    wls = np.mean(np.abs(pred[m] - yq[m]))
    wls_mape = np.mean(np.abs((yq[m] - pred[m]) / yq[m])) * 100
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


def load_baselines(cfg, eval_set, rebuild=False, n_jobs=None, rho=0.0, legacy=True):
    # baselines are context-dependent (SSVI refit sees the noisy quotes), so they're rho-aware
    # the same way load_eval is: rho=0 is the shared cache, anything else gets its own file
    stem = "noisy_baselines" if rho == 0.0 else f"noisy_baselines_{_rho_tag(rho)}"
    path = EVAL_DIR / f"{stem}{_legacy_tag(legacy)}.json"
    if path.exists() and not rebuild:
        raw = json.load(open(path))
        return {float(m): {int(n): v for n, v in rows.items()} for m, rows in raw.items()}
    base = _baselines(eval_set, cfg, n_jobs=n_jobs)
    json.dump(base, open(path, "w"), indent=2, default=str)
    return base


def _write_eval_txt(path, run_name, which, rho, results, arb_results, uq_results, baselines):
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

    lines.append("\n\nUncertainty (predictive-distribution calibration, see eval_uncertainty)")
    for m, rows in uq_results.items():
        lines.append(f"\n=== regime m={m} ===")
        lines.append(f"{'n_ctx':>6} {'CRPS':>8} {'pinball05':>10} {'pinball95':>10} {'width90%':>9}")
        for n_ctx, r in rows.items():
            lines.append(f"{n_ctx:>6} {r['crps']:>8.4f} {r['pinball'][0.05]:>10.4f}"
                         f" {r['pinball'][0.95]:>10.4f} {r['mean_interval_width']:>9.4f}")

    open(path, "w").write("\n".join(lines) + "\n")


def run_eval(run_name, cfg, device, rebuild_eval=False, which="final", rho=0.0, baseline_jobs=None,
             model_version=None, y_mean=0.300, y_scale=0.135, feature_scale=None, legacy=False):
    # matched-rho eval: a model trained on correlated quote noise (rho!=0) should be evaluated
    # against noise drawn the same way, not the default iid (rho=0) set - see notes/results_summary.md
    eval_set = load_eval(cfg, rebuild=rebuild_eval, rho=rho, legacy=legacy)
    arb_rows = load_arb_grid(cfg, rebuild=rebuild_eval)
    model, state = load_finetuned(run_name, device, which=which, model_version=model_version)
    results, arb_results, uq_results, pit_arrays = {}, {}, {}, {}
    slots = [(m, n_ctx) for m, by_ctx in eval_set.items() for n_ctx in by_ctx]
    for i, (m, n_ctx) in enumerate(slots, 1):
        tr, te = eval_set[m][n_ctx]
        print(f"[{i}/{len(slots)}] regime={m} n_ctx={n_ctx} ...", flush=True)
        mae, mape, cal, bf = eval_surfaces(model, tr, te, cfg, reload_state=state, model_version=model_version,
                                            iv_max=cfg["iv_max"], y_mean=y_mean, y_scale=y_scale,
                                            feature_scale=feature_scale)
        results.setdefault(str(m), {})[n_ctx] = dict(
            mae=float(mae), mape=float(mape), cal_viol=float(cal), bf_viol=float(bf))

        cell_frac, mean_depth, worst_cell, arb_free = eval_arbitrage_fine(
            model, tr, cfg, arb_rows, reload_state=state, model_version=model_version, iv_max=cfg["iv_max"],
            y_mean=y_mean, y_scale=y_scale, feature_scale=feature_scale)
        arb_results.setdefault(str(m), {})[n_ctx] = dict(
            cell_frac=float(cell_frac), mean_depth=float(mean_depth),
            worst_cell=float(worst_cell), arb_free=float(arb_free))

        uq = eval_uncertainty(model, tr, te, reload_state=state, model_version=model_version,
                               iv_max=cfg["iv_max"], y_mean=y_mean, y_scale=y_scale, feature_scale=feature_scale)
        uq_results.setdefault(str(m), {})[n_ctx] = dict(
            crps=uq["crps"], pinball=uq["pinball"], mean_interval_width=uq["mean_interval_width"])
        pit_arrays[f"{m}_{n_ctx}"] = uq["pit"]

    baselines = load_baselines(cfg, eval_set, rebuild=rebuild_eval, n_jobs=baseline_jobs, rho=rho, legacy=legacy)

    out_dir = ROOT / "checkpoints" / run_name
    json.dump(dict(mae=results, arb=arb_results, uq=uq_results), open(out_dir / "eval.json", "w"), indent=2)
    np.savez(out_dir / "pit.npz", **pit_arrays)
    _write_eval_txt(out_dir / "eval.txt", run_name, which, rho, results, arb_results, uq_results, baselines)
    print(f"\nsaved eval -> {out_dir/'eval.json'}, {out_dir/'eval.txt'}, and {out_dir/'pit.npz'} (PIT histogram data)")
    print(open(out_dir / "eval.txt").read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=list(EXPERIMENTS))
    p.add_argument("--run-name", default=None)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n-surfaces", type=int, default=200)
    p.add_argument("--n-context", type=int, nargs=2, default=(3, 60), metavar=("LO", "HI"))
    p.add_argument("--rho", type=float, nargs="+", default=[0.0], metavar="RHO",
                    help="quote-noise correlation (0=iid per quote, 1=shared per surface); one value "
                         "fixes it, two values (LO HI) sample it uniformly per surface; non-default "
                         "gets its own val/eval cache, the shared caches stay at rho=0")
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
    p.add_argument("--from-scratch", action="store_true",
                    help="randomize the pretrained TabPFN weights before training instead of "
                         "finetuning from them - needs a much higher --lr than the finetuning default")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--init-from", default=None,
                    help="continue training from checkpoints/<run>/final.pt instead of the "
                         "pretrained TabPFN weights (a fresh run, not a resume - new run_name/dir)")
    p.add_argument("--model-version", default=None, choices=["v2", "v2.5", "v2.6", "v3"],
                    help="TabPFN checkpoint version (default: current package default, v3). "
                         "Must already be cached (~/.cache/tabpfn) before running on a compute "
                         "node with no internet - see scripts/predownload_tabpfn.py")
    p.add_argument("--which", default="final", choices=["final", "best"],
                    help="checkpoint to evaluate with --eval-only")
    p.add_argument("--feature-zscore", action="store_true",
                    help="also z-score z/tau input features globally (cfg's z_mean/z_scale/"
                         "tau_mean/tau_scale) - off by default, features are fed raw")
    p.add_argument("--global-squashing", action="store_true",
                    help="robust median/IQR global rescale for y AND z/tau (cfg's *_median/*_iqr) "
                         "instead of mean/std - no clip (z/tau are already domain-bounded)")
    p.add_argument("--crps-only", action="store_true",
                    help="drop the default loss's mse_loss_weight=1.0 term (mean-only, no "
                         "calibration signal) and train on crps_loss_weight=1.0 alone - tests "
                         "whether the MSE term was diluting CRPS's pressure toward a calibrated "
                         "predictive distribution (see report_notes.md PIT investigation). Only "
                         "applies to the supervised experiment (arb already uses its own loss_fn).")
    p.add_argument("--heston-frac", type=float, default=None,
                    help="override cfg's mixture.heston_frac in-memory (0.0-1.0) - lets "
                         "concurrent runs use different fractions without editing the shared, "
                         "synced config.yaml on disk")
    p.add_argument("--calibrated-noise", action="store_true",
                    help="use config.yaml's calibrated (SPXW-fit) noise: block instead of the "
                         "default noise_legacy: constants, and route eval/val/baseline caches to "
                         "the calibrated files instead of the frozen '_oldnoise' ones. DEFAULT IS "
                         "LEGACY NOISE for now - the calibration is still being validated (see "
                         "notes/results_summary.md), so every run stays comparable to the existing "
                         "results unless this is passed explicitly")
    args = p.parse_args()

    spec = EXPERIMENTS[args.experiment]
    run_name = args.run_name or f"{args.experiment}_{args.n_context[0]}_{args.n_context[1]}"
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    legacy_noise = not args.calibrated_noise
    if legacy_noise:
        cfg["noise"] = cfg["noise_legacy"]
    if args.heston_frac is not None:
        cfg.setdefault("mixture", {})["heston_frac"] = args.heston_frac

    rho = args.rho[0] if len(args.rho) == 1 else tuple(args.rho)
    y_mean, y_scale = cfg["y_mean"], cfg["y_scale"]
    feature_scale = None
    if args.global_squashing:
        y_mean, y_scale = cfg["y_median"], cfg["y_iqr"]
        feature_scale = (cfg["z_median"], cfg["z_iqr"], cfg["tau_median"], cfg["tau_iqr"])
    elif args.feature_zscore:
        feature_scale = (cfg["z_mean"], cfg["z_scale"], cfg["tau_mean"], cfg["tau_scale"])

    if not args.eval_only:
        val_data = load_val(args.experiment, cfg, rebuild=args.rebuild_val, rho=rho, legacy=legacy_noise)
        data_provider = spec["provider"](cfg, tuple(args.n_context), rho)
        loss_fn = crps_only_loss if args.crps_only else spec["loss"](cfg)
        init_state = None
        if args.init_from is not None:
            init_state = torch.load(ROOT / "checkpoints" / args.init_from / "final.pt", map_location=args.device)
        finetune(
            data_provider, run_name=run_name, n_epochs=args.epochs,
            n_surfaces_per_epoch=args.n_surfaces, batch_size=args.batch_size or spec["batch_size"],
            group_size=args.group_size or spec["group_size"], val_group_size=spec["val_group_size"],
            val_data=val_data, val_every=args.val_every, iv_max=cfg["iv_max"],
            y_mean=y_mean, y_scale=y_scale, feature_scale=feature_scale,
            loss_fn=loss_fn, device=args.device,
            wandb_project=args.wandb_project or None, wandb_entity=args.wandb_entity,
            from_scratch=args.from_scratch, lr=args.lr, init_state=init_state,
            model_version=args.model_version,
        )

    run_eval(run_name, cfg, args.device, rebuild_eval=args.rebuild_eval, which=args.which,
             baseline_jobs=args.baseline_jobs, rho=rho, model_version=args.model_version,
             y_mean=y_mean, y_scale=y_scale, feature_scale=feature_scale, legacy=legacy_noise)


if __name__ == "__main__":
    main()
