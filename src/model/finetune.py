
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
from torch.utils.data import DataLoader

from tabpfn import TabPFNRegressor
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks, meta_dataset_collator
from tabpfn.finetuning.finetuned_regressor import _compute_regression_loss
from tabpfn.finetuning.train_util import get_cosine_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DataProvider = Callable[[int], tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]]


def _fixed_context_split_fn(n_context: int):
    def split_fn(X: np.ndarray, y: np.ndarray, stratify=None):
        return X[:n_context], X[n_context:], y[:n_context], y[n_context:]
    return split_fn


def _build_dataloader(estimator, train, test, n_context, *, seed, rng):
    X_list = [np.concatenate([X_tr, X_te]) for (X_tr, _), (X_te, _) in zip(train, test)]
    y_list = [np.concatenate([y_tr, y_te]) for (_, y_tr), (_, y_te) in zip(train, test)]

    datasets = get_preprocessed_dataset_chunks(
        calling_instance=estimator,
        X_raw=X_list,
        y_raw=y_list,
        split_fn=_fixed_context_split_fn(n_context),
        max_data_size=None,  # one dataset = one chunk; no intra-dataset splitting
        model_type="regressor",
        equal_split_size=False,
        data_shuffle_seed=seed,
        preprocessing_random_state=rng,
        shuffle=False,
    )
    return DataLoader(datasets, batch_size=1, collate_fn=meta_dataset_collator, shuffle=True)


def _run_batches(estimator, dataloader, perf_opts, device, *, optimizer=None, scheduler=None, grad_clip=None):
    """One pass over `dataloader`. Trains (backprop + step) if `optimizer` is given,
    otherwise runs a no-grad validation pass. Returns the mean loss."""
    training = optimizer is not None
    estimator.model_.train(training)

    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for batch in dataloader:
            if training:
                optimizer.zero_grad()

            # each dataset has its own target statistics for the bar distribution
            estimator.raw_space_bardist_ = batch.raw_space_bardist
            estimator.znorm_space_bardist_ = batch.znorm_space_bardist
            bardist_loss_fn = batch.znorm_space_bardist

            # cache context in executor - no gradient flows through this
            estimator.fit_from_preprocessed(
                batch.X_context,
                batch.y_context,
                batch.cat_indices,
                batch.configs,
                performance_options=perf_opts,
                no_refit=True,
            )

            # differentiable forward on query points
            _, per_estim_logits, _ = estimator.forward(batch.X_query, use_inference_mode=False)

            # list of [Q, B, L] per estimator -> [B*E, Q, L]
            logits_QBEL = torch.stack(per_estim_logits, dim=2)
            Q, B, E, L = logits_QBEL.shape
            logits_BQL = logits_QBEL.permute(1, 2, 0, 3).reshape(B * E, Q, L)
            targets_BQ = batch.y_query.repeat(B * E, 1).to(device)

            loss = _compute_regression_loss(
                logits_BQL=logits_BQL,
                targets_BQ=targets_BQ,
                bardist_loss_fn=bardist_loss_fn,
                ce_loss_weight=0.0,
                crps_loss_weight=1.0,
                mse_loss_weight=1.0,
            )

            if training:
                loss.backward()
                clip_grad_norm_(estimator.model_.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()

    estimator.model_.train()
    return total_loss / max(len(dataloader), 1)


def finetune(
    data_provider: DataProvider,
    *,
    n_epochs: int = 50,
    n_surfaces_per_epoch: int = 200,
    n_val_surfaces: int = 5,
    lr: float = 1e-5,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    warmup_ratio: float = 0.1,
    output_dir: str = "checkpoints/finetune_v1",
    device: str = "cpu",
    seed: int = 0,
) -> TabPFNRegressor:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    total_steps = n_epochs * n_surfaces_per_epoch
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    schedule_fn = get_cosine_schedule_with_warmup(total_steps=total_steps, warmup_steps=warmup_steps)
    scheduler = LambdaLR(optimizer, lr_lambda=schedule_fn)

    # fixed held-out surfaces, generated once, to track validation loss across epochs
    val_train, val_test = data_provider(n_val_surfaces) if n_val_surfaces > 0 else (None, None)

    log.info(
        "Starting finetuning: %d epochs x %d datasets (%d val), warmup_steps=%d/%d",
        n_epochs, n_surfaces_per_epoch, n_val_surfaces, warmup_steps, total_steps,
    )

    for epoch in range(n_epochs):
        train, test = data_provider(n_surfaces_per_epoch)
        n_context = len(train[0][0])

        dataloader = _build_dataloader(estimator, train, test, n_context, seed=seed + epoch, rng=rng)
        train_loss = _run_batches(
            estimator, dataloader, perf_opts, device,
            optimizer=optimizer, scheduler=scheduler, grad_clip=grad_clip,
        )

        if val_train is not None:
            val_dataloader = _build_dataloader(estimator, val_train, val_test, n_context, seed=seed, rng=rng)
            val_loss = _run_batches(estimator, val_dataloader, perf_opts, device)
            log.info("Epoch %d/%d | train_loss=%.4f val_loss=%.4f", epoch + 1, n_epochs, train_loss, val_loss)
        else:
            log.info("Epoch %d/%d | train_loss=%.4f", epoch + 1, n_epochs, train_loss)

        torch.save(estimator.model_.state_dict(), out / f"epoch_{epoch + 1:03d}.pt")

    torch.save(estimator.model_.state_dict(), out / "final.pt")
    log.info("Saved final model to %s/final.pt", out)
    return estimator


if __name__ == "__main__":
    import yaml
    from functools import partial
    from src.data_generation.data_preperation import data_preparation

    cfg = yaml.safe_load(open("config.yaml"))
    data_provider = partial(data_preparation, cfg, n_context=10)
    finetune(data_provider, n_epochs=1, n_surfaces_per_epoch=2, n_val_surfaces=2)
