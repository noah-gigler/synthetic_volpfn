# Builds TabPFN RegressorBatches from already-split context/query arrays
# skipping tabpfn.finetuning.data_util's generic split_fn/chunking machinery since we don't need it. 
# Preprocessing-config selection is based on context size alone

import numpy as np
import torch

from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
from tabpfn.finetuning.data_util import RegressorBatch
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor


def build_regression_batches(estimator, train, test, rng: np.random.Generator) -> list[RegressorBatch]:
    """One RegressorBatch per (context, query) surface, each with its own context size.

    `train`/`test` are the lists returned by a `data_provider`, i.e.
    `list[(X_context, y_context)]` and `list[(X_query, y_query)]`.
    """
    if not hasattr(estimator, "models_") or estimator.models_ is None:
        estimator._initialize_model_variables()

    batches = []
    for (X_context, y_context), (X_query_raw, y_query_raw) in zip(train, test):
        ensemble_configs, X_context, y_context, znorm_bardist = estimator._initialize_dataset_preprocessing(
            X=X_context, y=y_context, random_state=rng,
        )

        train_mean, train_std = np.mean(y_context), max(np.std(y_context), 1e-8)
        y_context_znorm = (y_context - train_mean) / train_std
        y_query_znorm = (y_query_raw - train_mean) / train_std
        raw_bardist = FullSupportBarDistribution(znorm_bardist.borders * train_std + train_mean).float()

        preprocessor = TabPFNEnsemblePreprocessor(
            configs=ensemble_configs,
            n_samples=X_context.shape[0],
            feature_schema=estimator.inferred_feature_schema_,
            random_state=rng,
            n_preprocessing_jobs=1,
        )
        members = preprocessor.fit_transform_ensemble_members(X_train=X_context, y_train=y_context_znorm)

        def batched(x, dtype=torch.float32):
            return torch.as_tensor(x, dtype=dtype).unsqueeze(0)

        batches.append(
            RegressorBatch(
                X_context=[batched(m.X_train) for m in members],
                X_query=[batched(m.transform_X_test(X_query_raw)) for m in members],
                y_context=[batched(m.y_train) for m in members],
                y_query=batched(y_query_znorm),
                cat_indices=[[m.feature_schema.indices_for(FeatureModality.CATEGORICAL) for m in members]],
                configs=[list(ensemble_configs)],
                raw_space_bardist=raw_bardist,
                znorm_space_bardist=znorm_bardist,
                X_query_raw=batched(X_query_raw),
                y_query_raw=batched(y_query_raw),
            )
        )

    return batches
