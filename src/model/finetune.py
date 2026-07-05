
# Meta-learning finetuning loop for TabPFN, data-agnostic.
#   1. Preprocess via TabPFN's pipeline (standardization, ensemble preprocessing)
#   2. Cache context in TabPFN executor (no gradient)
#   3. Differentiable forward on query -> bar distribution logits
#   4. CRPS + MSE loss on bar distribution -> backprop -> AdamW step

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from tabpfn import TabPFNRegressor
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.finetuning.finetuned_regressor import _compute_regression_loss
from tabpfn.finetuning.train_util import get_cosine_schedule_with_warmup

from src.model.preprocessed_dataset import preprocess_surfaces

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DataProvider = Callable[[int], tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]]

# Resolved from this file's location, not cwd, so checkpoints always land in
# <repo_root>/checkpoints/<run_name> regardless of where finetune() is called from
# (e.g. a notebook running with cwd=notebooks/).
_CHECKPOINTS_DIR = Path(__file__).resolve().parents[2] / "checkpoints"


def _run_pass(estimator, surfaces, perf_opts, device, *, optimizer=None, scheduler=None,
              grad_clip=None, batch_size=1):
    """One pass over `surfaces`. Trains if `optimizer` is given, accumulating gradients
    over `batch_size` surfaces per optimizer step (so one degenerate surface can't
    yank the weights alone); otherwise runs a no-grad validation pass.
    Returns per-surface losses in input order."""
    training = optimizer is not None
    estimator.model_.train(training)

    losses = []
    group_size = 1
    with torch.set_grad_enabled(training):
        for i, surface in enumerate(surfaces):
            if training and i % batch_size == 0:
                optimizer.zero_grad()
                group_size = min(batch_size, len(surfaces) - i)

            # each dataset has its own target statistics for the bar distribution
            estimator.raw_space_bardist_ = surface.raw_space_bardist
            estimator.znorm_space_bardist_ = surface.znorm_space_bardist
            bardist_loss_fn = surface.znorm_space_bardist

            # cache context in executor - no gradient flows through this
            estimator.fit_from_preprocessed(
                surface.X_context,
                surface.y_context,
                surface.cat_indices,
                surface.configs,
                performance_options=perf_opts,
                no_refit=True,
            )

            # differentiable forward on query points
            _, per_estim_logits, _ = estimator.forward(surface.X_query, use_inference_mode=False)

            # list of [Q, B, L] per estimator -> [B*E, Q, L]
            logits_QBEL = torch.stack(per_estim_logits, dim=2)
            Q, B, E, L = logits_QBEL.shape
            logits_BQL = logits_QBEL.permute(1, 2, 0, 3).reshape(B * E, Q, L)
            targets_BQ = surface.y_query.repeat(B * E, 1).to(device)

            loss = _compute_regression_loss(
                logits_BQL=logits_BQL,
                targets_BQ=targets_BQ,
                bardist_loss_fn=bardist_loss_fn,
                ce_loss_weight=0.0,
                crps_loss_weight=1.0,
                mse_loss_weight=1.0,
            )

            if training:
                (loss / group_size).backward()
                if (i + 1) % batch_size == 0 or i == len(surfaces) - 1:
                    clip_grad_norm_(estimator.model_.parameters(), grad_clip)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

            losses.append(loss.item())

    estimator.model_.train()
    return losses


def finetune(
    data_provider: DataProvider,
    run_name: str,
    *,
    n_epochs: int = 50,
    n_surfaces_per_epoch: int = 200,
    batch_size: int = 1,
    n_val_surfaces: int = 5,
    val_data: tuple[list, list] | None = None,
    val_every: int = 1,
    lr: float = 1e-5,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    warmup_ratio: float = 0.1,
    device: str = "cpu",
    seed: int = 0,
) -> TabPFNRegressor:
    out = _CHECKPOINTS_DIR / run_name
    out.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    estimator = TabPFNRegressor(
        fit_mode="batched",
        n_estimators=1,
        device=device,
        inference_config={"FINGERPRINT_FEATURE": False},
    )
    estimator._initialize_model_variables()
    estimator.model_.to(device)
    estimator.model_.train()

    optimizer = AdamW(estimator.model_.parameters(), lr=lr, weight_decay=weight_decay)
    perf_opts = PerformanceOptions(force_recompute_layer=False, use_chunkwise_inference=False)
    rng = np.random.default_rng(seed)

    # one optimizer step per batch of `batch_size` surfaces (last batch may be smaller)
    total_steps = n_epochs * -(-n_surfaces_per_epoch // batch_size)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    schedule_fn = get_cosine_schedule_with_warmup(total_steps=total_steps, warmup_steps=warmup_steps)
    scheduler = LambdaLR(optimizer, lr_lambda=schedule_fn)

    # fixed held-out (context, query) sets to track validation loss across epochs;
    # `val_data` (e.g. from make_stratified_eval_set) takes precedence over drawing
    # random surfaces from the data_provider. Val surfaces are preprocessed once:
    # redoing it each epoch would re-run preprocessing and re-consume RNG, so the
    # val task itself would drift between epochs.
    if val_data is not None:
        val_train, val_test = val_data
    elif n_val_surfaces > 0:
        val_train, val_test = data_provider(n_val_surfaces)
    else:
        val_train, val_test = None, None

    val_surfaces = None
    if val_train is not None:
        val_surfaces = preprocess_surfaces(estimator, val_train, val_test, rng)
        val_sizes = [len(y_ctx) for _, y_ctx in val_train]

    log.info(
        "Starting finetuning: %d epochs x %d datasets, batch_size=%d (%d val surfaces every %d epochs), warmup_steps=%d/%d",
        n_epochs, n_surfaces_per_epoch, batch_size, len(val_surfaces or []), val_every, warmup_steps, total_steps,
    )

    for epoch in range(n_epochs):
        train, test = data_provider(n_surfaces_per_epoch)

        surfaces = preprocess_surfaces(estimator, train, test, rng)
        rng.shuffle(surfaces)
        train_losses = _run_pass(
            estimator, surfaces, perf_opts, device,
            optimizer=optimizer, scheduler=scheduler, grad_clip=grad_clip,
            batch_size=batch_size,
        )
        train_loss = float(np.mean(train_losses))

        is_val_epoch = val_surfaces is not None and ((epoch + 1) % val_every == 0 or epoch == n_epochs - 1)
        if is_val_epoch:
            val_losses = _run_pass(estimator, val_surfaces, perf_opts, device)
            val_loss = float(np.mean(val_losses))

            by_size: dict[int, list[float]] = {}
            for size, surface_loss in zip(val_sizes, val_losses):
                by_size.setdefault(size, []).append(surface_loss)
            breakdown = " ".join(f"{s}={np.mean(v):.4f}" for s, v in sorted(by_size.items()))
            log.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f | by n_ctx: %s",
                epoch + 1, n_epochs, train_loss, val_loss, breakdown,
            )
        else:
            # no comparable val loss this epoch; only track best via train loss when
            # there is no val set at all
            val_loss = train_loss if val_surfaces is None else None
            log.info("Epoch %d/%d | train_loss=%.4f", epoch + 1, n_epochs, train_loss)

        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            torch.save(estimator.model_.state_dict(), out / "best.pt")
            log.info("Saved new best checkpoint (loss=%.4f) to %s/best.pt", best_loss, out)

    torch.save(estimator.model_.state_dict(), out / "final.pt")
    log.info("Saved final model to %s/final.pt", out)
    return estimator


if __name__ == "__main__":
    import yaml
    from functools import partial
    from src.data_generation.data_preperation import data_preparation, make_stratified_eval_set

    cfg = yaml.safe_load(open("config.yaml"))
    data_provider = partial(data_preparation, cfg, n_context=(3, 30))
    val_data = make_stratified_eval_set(cfg, n_surfaces=2, context_sizes=[3, 40])
    # 3 surfaces with batch_size=2 exercises the partial last batch
    finetune(data_provider, "smoke_test", n_epochs=1, n_surfaces_per_epoch=3, batch_size=2, val_data=val_data)
