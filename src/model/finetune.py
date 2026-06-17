"""
Meta-learning finetuning loop for TabPFN on synthetic SSVI vol surfaces.

Each gradient step processes one synthetic surface:
  1. Sample a full SSVI surface (n_k * n_ttm grid points)
  2. Split into context (Gaussian ATM-weighted) and query (remaining points)
  3. Preprocess via TabPFN's pipeline (standardization, ensemble preprocessing)
  4. Cache context in TabPFN executor (no gradient)
  5. Differentiable forward on query -> bar distribution logits
  6. Cross-entropy loss on bar distribution -> backprop -> AdamW step

Run:
    python -m src.model.finetune
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tabpfn import TabPFNRegressor
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.constants import ModelVersion
from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks, meta_dataset_collator
from tabpfn.finetuning.finetuned_regressor import _compute_regression_loss

from src.data_generation.data_preperation import generate_surfaces

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _context_query_split(X: np.ndarray, y: np.ndarray, *, n_context: int, rng: np.random.Generator, stratify=None):
    """Gaussian ATM-weighted context sampling; returns (X_ctx, X_query, y_ctx, y_query)."""
    k_weights = np.exp(-0.5 * (X[:, 0] / 0.25) ** 2)
    k_weights /= k_weights.sum()
    ctx_idx = rng.choice(len(X), size=n_context, replace=False, p=k_weights)
    query_idx = np.setdiff1d(np.arange(len(X)), ctx_idx)
    return X[ctx_idx], X[query_idx], y[ctx_idx], y[query_idx]


def _build_surface_lists(cfg: dict, n_surfaces: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate n_surfaces SSVI surfaces as flat [k, tau] -> implied_vol arrays."""
    ttms, ks, surfaces = generate_surfaces(cfg, n_surfaces)
    TT, KK = np.meshgrid(ttms, ks, indexing="ij")
    X_flat = np.column_stack([KK.ravel(), TT.ravel()]).astype(np.float32)

    X_list = [X_flat.copy() for _ in range(n_surfaces)]
    y_list = [surfaces[i].ravel().astype(np.float32) for i in range(n_surfaces)]
    return X_list, y_list


def finetune(
    cfg: dict,
    *,
    n_epochs: int = 50,
    n_surfaces_per_epoch: int = 200,
    n_context: int = 10,
    lr: float = 1e-5,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    output_dir: str = "checkpoints/finetune_v1",
    device: str = "cpu",
    seed: int = 42,
) -> TabPFNRegressor:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- model setup ----------
    estimator = TabPFNRegressor.create_default_for_version(
        version=ModelVersion.V3,
        fit_mode="batched",
        n_estimators=1,
        device=device,
    )
    estimator._initialize_model_variables()
    estimator.model_.to(device)
    estimator.model_.train()

    optimizer = AdamW(estimator.model_.parameters(), lr=lr, weight_decay=weight_decay)
    perf_opts = PerformanceOptions(force_recompute_layer=False, use_chunkwise_inference=False)
    rng = np.random.default_rng(seed)

    log.info("Starting finetuning: %d epochs x %d surfaces, n_context=%d", n_epochs, n_surfaces_per_epoch, n_context)

    for epoch in range(n_epochs):
        X_list, y_list = _build_surface_lists(cfg, n_surfaces_per_epoch)

        split_fn = partial(_context_query_split, n_context=n_context, rng=rng)

        training_datasets = get_preprocessed_dataset_chunks(
            calling_instance=estimator,
            X_raw=X_list,
            y_raw=y_list,
            split_fn=split_fn,
            max_data_size=None,   # one surface = one dataset chunk; no intra-surface splitting
            model_type="regressor",
            equal_split_size=False,
            data_shuffle_seed=seed + epoch,
            preprocessing_random_state=rng,
            shuffle=False,
        )

        dataloader = DataLoader(
            training_datasets,
            batch_size=1,
            collate_fn=meta_dataset_collator,
            shuffle=True,
        )

        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()

            # Register bar distribution for this surface's target statistics
            estimator.raw_space_bardist_ = batch.raw_space_bardist
            estimator.znorm_space_bardist_ = batch.znorm_space_bardist
            bardist_loss_fn = batch.znorm_space_bardist

            # Cache context in executor (no gradient flows through here)
            estimator.fit_from_preprocessed(
                batch.X_context,
                batch.y_context,
                batch.cat_indices,
                batch.configs,
                performance_options=perf_opts,
                no_refit=True,
            )

            # Differentiable forward on query points
            _, per_estim_logits, _ = estimator.forward(batch.X_query, use_inference_mode=False)

            # Reshape: list of [Q, B, L] -> [B*E, Q, L]
            logits_QBEL = torch.stack(per_estim_logits, dim=2)
            Q, B, E, L = logits_QBEL.shape
            logits_BQL = logits_QBEL.permute(1, 2, 0, 3).reshape(B * E, Q, L)
            targets_BQ = batch.y_query.repeat(B * E, 1).to(device)

            loss = _compute_regression_loss(
                logits_BQL=logits_BQL,
                targets_BQ=targets_BQ,
                bardist_loss_fn=bardist_loss_fn,
                ce_loss_weight=1.0,
            )

            loss.backward()
            clip_grad_norm_(estimator.model_.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()

        mean_loss = epoch_loss / max(len(dataloader), 1)
        log.info("Epoch %d/%d | loss=%.4f", epoch + 1, n_epochs, mean_loss)

        torch.save(estimator.model_.state_dict(), out / f"epoch_{epoch + 1:03d}.pt")

    torch.save(estimator.model_.state_dict(), out / "final.pt")
    log.info("Saved final model to %s/final.pt", out)
    return estimator


if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    finetune(cfg, n_epochs=1, n_surfaces_per_epoch=2, n_context=10)
