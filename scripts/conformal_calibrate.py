# Post-hoc split-conformal (CQR) recalibration: fit an additive interval correction per
# n_ctx on a dedicated calibration set (built fresh here, not the shared 128-surface val
# cache training uses for checkpoint selection - keeping this separate avoids touching
# infrastructure other runs depend on), apply it to the existing eval set, report before/after
# coverage - no retraining. See report_notes.md's PIT investigation for why: the model is
# overconfident at small n_ctx and increasingly underconfident at large n_ctx.
#
# Calibration surfaces, not their individual grid points, are conformal prediction's
# exchangeable unit - the ~375 grid points on one surface are correlated (a model that's
# confidently wrong about one surface is usually wrong at many of its points at once, not
# 375 independent times), so the *surface* count, not the raw point count, sets the
# effective sample size for the quantile estimate. `--cal-n` should be generous per (regime,
# n_ctx) cell for that reason, not just "large" in raw point terms.
import argparse
from pathlib import Path

import numpy as np
import yaml

from scripts.run_finetuning import EVAL_CTX_SIZES, EVAL_REGIMES, load_eval, load_finetuned
from src.data_generation.noise import make_noisy_stratified_eval_set
from src.evaluation.surface_eval import conformal_correction, predict_interval

ROOT = Path(__file__).resolve().parents[1]


def _build_calibration_set(cfg, n_per_cell, ctx_sizes, seed=12345):
    # dedicated set, disjoint seed from both VAL_SEED and EVAL_SEED in run_finetuning.py.
    # kept regime-separated (list-of-lists, not concatenated) so a smaller prefix of `n`
    # surfaces per regime stays balanced across regimes - concatenating first would make any
    # prefix shorter than n_per_cell come entirely from the first regime's block
    slices = {n_ctx: [([], []) for _ in EVAL_REGIMES] for n_ctx in ctx_sizes}
    for r, m in enumerate(EVAL_REGIMES):
        np.random.seed(seed + int(m * 1000))
        tr, te = make_noisy_stratified_eval_set(cfg, n_per_cell, ctx_sizes, regime=m)
        for i, n_ctx in enumerate(ctx_sizes):
            slices[n_ctx][r][0].extend(tr[i * n_per_cell:(i + 1) * n_per_cell])
            slices[n_ctx][r][1].extend(te[i * n_per_cell:(i + 1) * n_per_cell])
    return slices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--which", default="final", choices=["final", "best"])
    p.add_argument("--alpha", type=float, default=0.10, help="1-alpha = target coverage, default 90%")
    p.add_argument("--cal-n-sweep", type=int, nargs="+", default=[64, 256, 1024, 4096],
                    help="calibration surfaces per (regime, n_ctx) cell, ascending - the model "
                         "forward pass runs once at the largest size; smaller sizes are nested "
                         "prefix subsets scored via cheap numpy re-slicing, not separate passes")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    cal_sizes = sorted(args.cal_n_sweep)
    n_max = cal_sizes[-1]

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    model, state = load_finetuned(args.run_name, args.device, which=args.which)

    cal_by_ctx = _build_calibration_set(cfg, n_max, EVAL_CTX_SIZES)
    eval_set = load_eval(cfg)

    print(f"{args.run_name} ({args.which}.pt) - conformal recalibration sweep, "
          f"target coverage {1 - args.alpha:.0%}, cal sizes (surfaces/regime/n_ctx) = {cal_sizes}\n")

    for n_ctx in EVAL_CTX_SIZES:
        # calibration side: one forward pass per regime at n_max surfaces, reshaped to
        # (n_regimes, n_max, n_points) so a size-n prefix ([:, :n, :]) stays balanced across
        # regimes instead of draining regime 0's block first - no extra forward passes needed
        cal_lo_r, cal_hi_r, cal_y_r = [], [], []
        for cal_tr, cal_te in cal_by_ctx[n_ctx]:
            lo, hi, y = predict_interval(model, cal_tr, cal_te, reload_state=state, iv_max=cfg["iv_max"])
            n_points = len(lo) // len(cal_tr)
            cal_lo_r.append(lo.reshape(len(cal_tr), n_points))
            cal_hi_r.append(hi.reshape(len(cal_tr), n_points))
            cal_y_r.append(y.reshape(len(cal_tr), n_points))
        cal_lo3, cal_hi3, cal_y3 = np.stack(cal_lo_r), np.stack(cal_hi_r), np.stack(cal_y_r)

        # eval side: one forward pass total, independent of cal_n, reused for every sweep point
        raw_cov, raw_w, eval_los, eval_his, eval_ys = [], [], [], [], []
        for m in EVAL_REGIMES:
            tr, te = eval_set[m][n_ctx]
            lo, hi, y = predict_interval(model, tr, te, reload_state=state, iv_max=cfg["iv_max"])
            raw_cov.append(np.mean((y >= lo) & (y <= hi)))
            raw_w.append(np.mean(hi - lo))
            eval_los.append(lo); eval_his.append(hi); eval_ys.append(y)

        print(f"=== n_ctx={n_ctx}  (raw cov {np.mean(raw_cov)*100:.1f}%, raw width {np.mean(raw_w):.4f}) ===")
        print(f"{'cal_n/regime':>13} {'Q':>8} {'corr cov%':>10} {'corr width':>11}")
        for n in cal_sizes:
            # nested prefix: first `n` surfaces per regime (already-drawn superset, not a
            # fresh draw), so every sweep point uses a strict, regime-balanced subset of the
            # largest one - no extra forward passes needed for the smaller sweep points
            lo_n, hi_n, y_n = cal_lo3[:, :n].ravel(), cal_hi3[:, :n].ravel(), cal_y3[:, :n].ravel()
            Q = conformal_correction(lo_n, hi_n, y_n, alpha=args.alpha)

            corr_cov, corr_w = [], []
            for lo, hi, y in zip(eval_los, eval_his, eval_ys):
                lo2, hi2 = lo - Q, hi + Q
                corr_cov.append(np.mean((y >= lo2) & (y <= hi2)))
                corr_w.append(np.mean(hi2 - lo2))

            print(f"{n:>13} {Q:>8.4f} {np.mean(corr_cov)*100:>9.1f}% {np.mean(corr_w):>11.4f}")
        print()


if __name__ == "__main__":
    main()
