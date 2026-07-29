# OpDS-protocol training on real SPXW: self-fit at full-surface scale, the setting
# Wiedemann/Jacquier/Gonon (ICLR 2025) actually train in. Separate from run_real.py, which
# trains sparse-context held-out interpolation (3-60 quotes in, 1700 held-out scored) - a
# different task. Nothing in the existing pipeline is modified; finetune(), quote_arb_loss,
# sample_arb_grid and preprocess_surfaces are all used as-is.
#
# Per chunk of `group_size` days: one shared arb grid, and n_quote = min(quotes) over the
# chunk. Every surface in the chunk then has the same query length, which is what lets them
# stack - quote_arb_loss reads its row boundaries off surface 0 and applies them to the whole
# group, so a per-surface grid would silently mis-slice surfaces 2..G.
#
# The min() is deliberately the *only* source of density variation. OpDS draw keep_prob ~
# U(0.6, 1.2) per snapshot; taking min over a random chunk reproduces that distribution closely
# for free (measured: per-surface retention p5/median/p95 = 0.73/0.92/1.00 at group 4 vs their
# 0.63/0.90/1.00), so no second randomization is layered on top.
#
# Subsampling is UNIFORM over quotes, not the ATM-Gaussian weighting _split_surface uses -
# OpDS's keep_mask is plain row dropout, and ATM weighting would preferentially drop the wings,
# which is exactly where arbitrage violations live.
#
#   bench:  uv run python -m scripts.run_opds_train bench
#   train:  uv run python -m scripts.run_opds_train train <run-name> [group_size] [epochs]
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from tabpfn import TabPFNRegressor
from tabpfn.architectures.interface import PerformanceOptions

from src.data_generation.grid import sample_arb_grid
from src.data_generation.noise import ASK, BID, TRUE
from src.model.finetune import finetune
from src.model.preprocessed_dataset import preprocess_surfaces
from src.model.quote_loss import quote_arb_loss
from src.real_data.dataloader import temporal_split

ROOT = Path(__file__).resolve().parents[1]


def opds_task(pool, n, cfg, size_group, cap=None):
    """n surfaces in chunks of `size_group`; one arb grid and one quote count per chunk.
    `cap` limits quotes per surface (curriculum); None = the chunk's natural min."""
    train, test = [], []
    for start in range(0, n, size_group):
        days = [pool[i] for i in np.random.randint(0, len(pool), min(size_group, n - start))]
        arb_rows = sample_arb_grid(cfg)
        n_q = min(len(d) for d in days)
        if cap is not None:
            n_q = max(3, min(n_q, int(cap)))
        for d in days:
            idx = np.random.choice(len(d), size=n_q, replace=False)   # uniform, not ATM-weighted
            z, tau = d["z"].values[idx], d["tau"].values[idx]
            bid, ask = d["bid_iv"].values[idx], d["ask_iv"].values[idx]
            X_ctx = np.column_stack([np.tile(z, 2), np.tile(tau, 2), np.repeat([BID, ASK], n_q)])
            y_ctx = np.concatenate([bid, ask])
            held = np.column_stack([z, tau, np.full(n_q, TRUE)])
            X_q = np.vstack([arb_rows, held])
            y_q = np.full((len(X_q), 2), np.nan)
            y_q[len(arb_rows):] = np.column_stack([bid, ask])
            train.append((X_ctx, y_ctx))
            test.append((X_q, y_q))
    return train, test


class Curriculum:
    """Stateful data_provider: ramps quotes/surface geometrically over training.

    finetune() calls data_provider(n) once per epoch with no epoch argument, so the schedule
    lives here. Rationale: the interval NLL only constrains the predicted VALUE once the
    predictive distribution is narrow enough to sit inside the spread. Handed 4,300 tight
    constraints from scratch the model cannot satisfy them, hedging caps its loss at ~3.5 while
    sharp-and-wrong costs 13.8, and it stays diffuse forever (measured: width90 ~24 half-spreads
    vs 0.7 for a sparse-trained model). Starting sparse makes sharpness cheap to acquire; the
    ramp then tightens the fit - and therefore the arbitrage pressure - gradually.
    """

    def __init__(self, pool, cfg, size_group, n_epochs, lo=8, hi=4000, ramp=0.5):
        self.pool, self.cfg, self.g = pool, cfg, size_group
        self.lo, self.hi, self.n_ramp = lo, hi, max(1, int(ramp * n_epochs))
        self.epoch = 0

    def cap_at(self, epoch):
        f = min(1.0, epoch / self.n_ramp)
        return int(round(self.lo * (self.hi / self.lo) ** f))      # geometric

    def __call__(self, n):
        cap = self.cap_at(self.epoch)
        self.epoch += 1
        return opds_task(self.pool, n, self.cfg, self.g, cap=cap)


def _estimator(device):
    est = TabPFNRegressor(fit_mode="batched", n_estimators=1, device=device,
                          categorical_features_indices=[],
                          inference_config={"FINGERPRINT_FEATURE": False})
    est._initialize_model_variables()
    est.model_.to(device)   # the constructor's device arg does not move the weights
    return est


def bench(group_sizes=(1, 2, 4, 8, 16, 32, 64)):
    """Time one fwd+bwd per group size and report peak VRAM, until OOM."""
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    train_pool, _, _ = temporal_split("2020-01-01", "2023-12-31", val_months=3, test_months=6, cfg=cfg)
    device = "cuda"
    est = _estimator(device)
    perf = PerformanceOptions(force_recompute_layer=False, use_chunkwise_inference=False)
    loss_fn = partial(quote_arb_loss, cfg=cfg, lambda_cal=10.0, lambda_bf=10.0,
                      lambda_reg_z=0.01, lambda_reg_r=0.01, return_parts=True)
    print(f"{'group':>6} {'ctx rows':>9} {'query rows':>11} {'s/group':>9} {'s/surface':>10} "
          f"{'peak VRAM':>10}   15k-step est.")
    for G in group_sizes:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            np.random.seed(0)
            tr, te = opds_task(train_pool, G, cfg, G)
            surfaces = preprocess_surfaces(est, tr, te, np.random.default_rng(0), cfg["iv_max"],
                                           group_size=G, y_mean=cfg["y_mean"], y_scale=cfg["y_scale"])
            s = surfaces[0]
            est.raw_space_bardist_, est.znorm_space_bardist_ = s.raw_space_bardist, s.znorm_space_bardist
            torch.cuda.synchronize(); t0 = time.time()
            est.fit_from_preprocessed(s.X_context, s.y_context, s.cat_indices, s.configs,
                                      performance_options=perf, no_refit=True)
            _, per_estim, _ = est.forward(s.X_query, use_inference_mode=False)
            q = torch.stack(per_estim, dim=2)
            Q, B, E, L = q.shape
            out = loss_fn(est, s, q.permute(1, 2, 0, 3).reshape(B * E, Q, L))
            loss = (out[0] if isinstance(out, tuple) else out).mean()
            loss.backward()
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            # 15k steps at batch_size == group_size -> 15k groups
            print(f"{G:>6} {s.X_context[0].shape[1]:>9} {s.X_query[0].shape[1]:>11} "
                  f"{dt:>9.2f} {dt/G:>10.3f} {peak:>9.1f}G   {dt*15000/3600:>5.1f}h")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"{G:>6}  OOM ({torch.cuda.max_memory_allocated()/2**30:.0f}G peak before failure)")
            break
        finally:
            est.model_.zero_grad(set_to_none=True)


def train(run_name, group_size=16, epochs=500, n_surfaces=480,
          y_source="synthetic", init_from=None, eps=0.0, lam_bf=10.0, lr=1e-5):
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    train_pool, val_pool, _ = temporal_split("2020-01-01", "2023-12-31", val_months=3,
                                             test_months=6, cfg=cfg)
    group_size, epochs, n_surfaces = int(group_size), int(epochs), int(n_surfaces)
    sfx = "_real" if y_source == "real" else ""
    y_mean, y_scale = cfg["y_mean" + sfx], cfg["y_scale" + sfx]
    # warm start MUST use the y-scaling its source was trained with, or the bar-distribution
    # bin edges shift under the loaded weights (see report_notes: warm-start collapse)
    init_state = None if init_from in (None, "none") else torch.load(
        ROOT / "checkpoints" / init_from / "final.pt", map_location="cpu")
    steps = epochs * -(-n_surfaces // group_size)   # finetune's total_steps
    print(f"OpDS-protocol training | {len(train_pool)} train days, {len(val_pool)} val | "
          f"group={group_size} == batch | {epochs} epochs x {n_surfaces} surfaces "
          f"= {steps:,} optimizer steps, {epochs*n_surfaces:,} surface-passes | "
          f"y={y_source} ({y_mean}/{y_scale}) | init={init_from or 'pretrained TabPFN'} "
          f"| eps={eps} lam_bf={lam_bf}")
    np.random.seed(0)
    val_data = opds_task(val_pool, 64, cfg, group_size)
    finetune(
        partial(opds_task, train_pool, cfg=cfg, size_group=group_size),
        run_name=run_name, n_epochs=epochs, n_surfaces_per_epoch=n_surfaces,
        batch_size=group_size, group_size=group_size, val_group_size=group_size,
        val_data=val_data, val_every=25, loss_fn=partial(
            quote_arb_loss, cfg=cfg, lambda_cal=float(lam_bf), lambda_bf=float(lam_bf),
            lambda_reg_z=0.01, lambda_reg_r=0.01,
            eps_bf=float(eps), eps_cal=float(eps), return_parts=True),
        iv_max=cfg["iv_max"], y_mean=y_mean, y_scale=y_scale, lr=lr,
        init_state=init_state, wandb_project="volpfn", wandb_entity="volpfn",
    )


def curriculum(run_name, group_size=4, epochs=800, n_surfaces=120, y_source="real",
               init_from="real_arb_selffit_yreal_72k", eps=1e-3, lam_bf=10.0, lr=1e-5):
    """The combined run: context curriculum + margin hinges + warm start.

    Three things today established, in one configuration:
      - cold-start at full context falls into a hedging equilibrium it never escapes, so the
        context is ramped (and warm-started from a model that is already sharp);
      - `mean(relu(-g))` stops pushing the moment a cell crosses zero, so cells park on the
        boundary and cross back - hence the margin `eps`;
      - raw mid quotes are themselves 16% butterfly-violating, so the target is NOT to fit them
        but to sit inside the spread while strictly arb-free.
    """
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    train_pool, val_pool, _ = temporal_split("2020-01-01", "2023-12-31", val_months=3,
                                             test_months=6, cfg=cfg)
    group_size, epochs, n_surfaces = int(group_size), int(epochs), int(n_surfaces)
    eps, lam_bf, lr = float(eps), float(lam_bf), float(lr)
    sfx = "_real" if y_source == "real" else ""
    y_mean, y_scale = cfg["y_mean" + sfx], cfg["y_scale" + sfx]
    init_state = None if init_from in (None, "none") else torch.load(
        ROOT / "checkpoints" / init_from / "final.pt", map_location="cpu")
    prov = Curriculum(train_pool, cfg, group_size, epochs)
    steps = epochs * -(-n_surfaces // group_size)
    print(f"CURRICULUM | {epochs} ep x {n_surfaces} surf, group=batch={group_size} "
          f"= {steps:,} steps | quotes/surface {prov.cap_at(0)} -> {prov.cap_at(epochs)} "
          f"(ramp over {prov.n_ramp} ep) | eps={eps} lam_bf={lam_bf} | init={init_from}")
    np.random.seed(0)
    val_data = opds_task(val_pool, 64, cfg, group_size)          # val always at FULL context
    finetune(
        prov, run_name=run_name, n_epochs=epochs, n_surfaces_per_epoch=n_surfaces,
        batch_size=group_size, group_size=group_size, val_group_size=group_size,
        val_data=val_data, val_every=25, loss_fn=partial(
            quote_arb_loss, cfg=cfg, lambda_cal=lam_bf, lambda_bf=lam_bf,
            lambda_reg_z=0.01, lambda_reg_r=0.01, eps_bf=eps, eps_cal=eps, return_parts=True),
        iv_max=cfg["iv_max"], y_mean=y_mean, y_scale=y_scale, lr=lr,
        init_state=init_state, wandb_project="volpfn", wandb_entity="volpfn",
    )


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["bench"]
    {"bench": bench, "train": train, "curriculum": curriculum}[cmd](*rest)
