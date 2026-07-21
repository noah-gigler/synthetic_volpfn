# Preprocesses already-split (context, query) surfaces into TabPFN RegressorBatch containers,
# skipping tabpfn.finetuning.data_util's generic split_fn/chunking machinery since we don't need it.
# Preprocessing-config selection is based on context size alone

from pathlib import Path

import numpy as np
import torch
import yaml

from tabpfn.architectures.base.bar_distribution import BarDistribution
from tabpfn.finetuning.data_util import RegressorBatch
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor

_cfg = yaml.safe_load(open(Path(__file__).resolve().parents[2] / "config.yaml"))
Y_MEAN, Y_SCALE = _cfg["y_mean"], _cfg["y_scale"]


def preprocess_surfaces(estimator, train, test, rng: np.random.Generator, iv_max: float, group_size: int = 1) -> list[RegressorBatch]:
    """One RegressorBatch per group of up to `group_size` consecutive surfaces with equal
    context shape (stacked along the dataset-batch dim -> one forward pass per group).

    The bar distribution lives in a global z-normed space (y - Y_MEAN) / Y_SCALE - same fixed
    range for every surface (unlike the old per-surface znorm), fixing a gradient-conditioning
    problem raw-IV targets had (values ~0.1-0.3 made Adam's eps non-negligible, degrading long
    runs - see notes/results_summary.md). `raw_bardist` is the same bucket structure with
    borders mapped back to real IV (`* Y_SCALE + Y_MEAN`), for callers that need real units
    (arb/butterfly physics, MAE/eval) - `.mean()`/`.icdf()` on it decode automatically.

    `train`/`test` are the lists returned by a `data_provider`, i.e.
    `list[(X_context, y_context)]` and `list[(X_query, y_query)]`.
    """
    if not hasattr(estimator, "models_") or estimator.models_ is None:
        estimator._initialize_model_variables()

    device = next(estimator.model_.parameters()).device
    n_bars = estimator.znorm_space_bardist_.borders.shape[0] - 1
    znorm_borders = torch.linspace((0.0 - Y_MEAN) / Y_SCALE, (iv_max - Y_MEAN) / Y_SCALE, n_bars + 1)
    znorm_bardist = BarDistribution(znorm_borders).float().to(device)
    raw_bardist = BarDistribution(znorm_borders * Y_SCALE + Y_MEAN).float().to(device)

    built = []
    for (X_context, y_context), (X_query_raw, y_query_raw) in zip(train, test):
        y_context = (y_context - Y_MEAN) / Y_SCALE
        ensemble_configs, X_context, y_context, _ = estimator._initialize_dataset_preprocessing(
            X=X_context, y=y_context, random_state=rng,
        )

        preprocessor = TabPFNEnsemblePreprocessor(
            configs=ensemble_configs,
            n_samples=X_context.shape[0],
            feature_schema=estimator.inferred_feature_schema_,
            random_state=rng,
            n_preprocessing_jobs=1,
        )
        members = preprocessor.fit_transform_ensemble_members(X_train=X_context, y_train=y_context)

        def t(x):
            return torch.as_tensor(x, dtype=torch.float32, device=device)

        # X_query_raw/y_query_raw stay on CPU - they're only ever consumed via .numpy()
        # (eval's raw-space MAE/arb math), never fed back into the model
        def t_cpu(x):
            return torch.as_tensor(x, dtype=torch.float32)

        built.append({
            "X_context": [t(m.X_train) for m in members],
            "X_query": [t(m.transform_X_test(X_query_raw)) for m in members],
            "y_context": [t(m.y_train) for m in members],
            "y_query": t((y_query_raw - Y_MEAN) / Y_SCALE),
            "cat_indices": [m.feature_schema.indices_for(FeatureModality.CATEGORICAL) for m in members],
            "configs": list(ensemble_configs),
            "raw_bardist": raw_bardist,
            "znorm_bardist": znorm_bardist,
            "X_query_raw": t_cpu(X_query_raw),
            "y_query_raw": t_cpu(y_query_raw),
        })

    groups, cur = [], [built[0]]
    for s in built[1:]:
        if len(cur) < group_size and _stackable(cur[0], s):
            cur.append(s)
        else:
            groups.append(cur)
            cur = [s]
    groups.append(cur)
    return [_stack_group(g) for g in groups]


def _stackable(a, b):
    return (all(x.shape == y.shape for x, y in zip(a["X_context"], b["X_context"]))
            and all(x.shape == y.shape for x, y in zip(a["X_query"], b["X_query"])))


def _stack_group(group) -> RegressorBatch:
    n_estimators = len(group[0]["X_context"])
    batch = RegressorBatch(
        X_context=[torch.stack([s["X_context"][e] for s in group]) for e in range(n_estimators)],
        X_query=[torch.stack([s["X_query"][e] for s in group]) for e in range(n_estimators)],
        y_context=[torch.stack([s["y_context"][e] for s in group]) for e in range(n_estimators)],
        y_query=torch.stack([s["y_query"] for s in group]),
        cat_indices=[s["cat_indices"] for s in group],
        configs=[s["configs"] for s in group],
        raw_space_bardist=group[0]["raw_bardist"],
        znorm_space_bardist=group[0]["znorm_bardist"],
        X_query_raw=torch.stack([s["X_query_raw"] for s in group]),
        y_query_raw=torch.stack([s["y_query_raw"] for s in group]),
    )
    # all surfaces share the same fixed bardist; kept as per-surface lists for the
    # loss indexing contract
    batch.raw_bardists = [s["raw_bardist"] for s in group]
    batch.znorm_bardists = [s["znorm_bardist"] for s in group]
    return batch
