# Arb-grid check using the OpDS-benchmark context: retained 50% of a surface's quotes (the same
# split scripts/run_opds_benchmark.py uses for the interpolation setting), instead of the sparse
# n_ctx context src/real_data eval_real_*.json uses. Answers "is the OpDS-benchmark low MAPE
# bought with worse structural (calendar/butterfly) arbitrage" directly, on the same context the
# MAPE numbers were computed from.
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.run_finetuning import load_arb_grid
from scripts.run_opds_benchmark import split_surface, DROP_FRAC, SEED
from src.data_generation.noise import BID, ASK
from src.evaluation.surface_eval import _get_eval_estimator, eval_arbitrage_fine
from src.real_data.dataloader import temporal_split

ROOT = Path(__file__).resolve().parents[1]


def run(run_name, which="final", y_source="real"):
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    sfx = "_real" if y_source == "real" else ""
    y_mean, y_scale = cfg["y_mean" + sfx], cfg["y_scale" + sfx]
    _, _, pool = temporal_split("2020-01-01", "2023-12-31", val_months=3, test_months=6, cfg=cfg)
    arb_rows = load_arb_grid(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    est, pretrained = _get_eval_estimator(None)
    state = pretrained if run_name == "pretrained" else torch.load(
        ROOT / "checkpoints" / run_name / f"{which}.pt", map_location=device)

    for mode in ("interpolation", "extrapolation"):
        rng = np.random.default_rng(SEED)
        train_list = []
        for s in pool:
            z, tau = s["z"].values, s["tau"].values
            bid, ask = s["bid_iv"].values, s["ask_iv"].values
            keep, _ = split_surface(s, mode, rng)
            nk = len(keep)
            X_ctx = np.column_stack([np.tile(z[keep], 2), np.tile(tau[keep], 2),
                                     np.repeat([BID, ASK], nk)])
            y_ctx = np.concatenate([bid[keep], ask[keep]])
            train_list.append((X_ctx, y_ctx))

        cell_frac, mean_depth, worst_cell, arb_free = eval_arbitrage_fine(
            est, train_list, cfg, arb_rows, reload_state=state, iv_max=cfg["iv_max"],
            y_mean=y_mean, y_scale=y_scale)
        print(f"{run_name} ({which}.pt) | {mode:14} | context={DROP_FRAC:.0%} retained, "
              f"{len(pool)} surfaces")
        print(f"  cell_frac={cell_frac*100:.2f}%  mean_depth={mean_depth:.4f}"
              f"  worst_cell={worst_cell:.4f}  arb_free={arb_free*100:.1f}%\n")


if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else "real_arb_selffit_yreal_72k"
    which = sys.argv[2] if len(sys.argv) > 2 else "final"
    y_source = sys.argv[3] if len(sys.argv) > 3 else "real"
    run(run_name, which, y_source)
