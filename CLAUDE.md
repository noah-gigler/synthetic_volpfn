# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Masters semester project studying whether TabPFN (a tabular foundation model) can be finetuned to
interpolate/extrapolate implied volatility surfaces from a sparse set of quoted points, using synthetic
SSVI-generated surfaces as training data (see `VolSmoothing_with_TabPFN_proposal.pdf`).

## Commands

- Environment is managed with `uv` (Python 3.12 pinned via `.python-version`); dependencies in `pyproject.toml` / `uv.lock`.
- Run a module: `uv run python -m src.data_generation.data_preperation` or `uv run python -m src.model.finetune`
- No test suite, linter, or build step currently exists in this repo.
- Notebooks are stripped of output on commit via `nbstripout` (declared dependency).

## Architecture

Data flow: `config.yaml` (SSVI prior + grid settings) → `src/data_generation` → `src/model/finetune.py` → `checkpoints/`.

- **`config.yaml`** — single source of truth for the SSVI parameter prior (median/sigma per parameter,
  sampled as lognormal/logit-normal) and the `(k, ttm)` grid shape. Both data generation and finetuning
  load this file directly.
- **`src/data_generation/SSVI.py`** — samples SSVI parameters from the prior and evaluates the SSVI
  parametrization on a `(ttm, k)` grid, vectorized over a batch of surfaces. Enforces butterfly-arbitrage
  bounds on `eta` (Gatheral & Jacquier 2013, Theorem 4.2) when sampling.
- **`src/data_generation/data_preperation.py`** — turns full surfaces into (context, query) point sets:
  `sample_sparse_points` draws a Gaussian ATM-weighted, uniform-in-ttm subset of grid indices as the
  "sparse quotes" context; `data_preparation` builds the corresponding train/test `(X=[k,tau], y=iv)` arrays
  per surface. Has a `__main__` smoke test that fits/predicts a single surface with vanilla `TabPFNRegressor`.
- **`src/model/preprocessed_dataset.py`** — `build_regression_batches(estimator, train, test, rng)` turns a
  `data_provider`'s pre-split `(context, query)` arrays into a list of TabPFN `RegressorBatch`es, one per
  surface. Deliberately *not* using TabPFN's own `tabpfn.finetuning.data_util.get_preprocessed_dataset_chunks`:
  that helper assumes one raw dataset that still needs a generic `split_fn`-based train/test split (plus lazy
  chunking for oversized datasets), neither of which applies here since context/query are already split and
  small. Preprocessing-config selection (`_initialize_dataset_preprocessing`, i.e. which pipelines/target
  transforms TabPFN picks) is run on **context only**, matching TabPFN's own `fit(X_train, y_train)` semantics
  (config selection never sees test data in normal use) and how a real deployment would work (only sparse
  quotes are ever available, never a full grid) — this differs from the old finetuning-helper pattern, which
  selected configs based on train+test combined, and is a deliberate choice, not an oversight. Surfaces in one
  epoch can have different context sizes; nothing here assumes a fixed `n_context`.
- **`src/model/finetune.py`** — the actual meta-learning finetuning loop, data-agnostic: it takes a
  `data_provider(n) -> (train, test)` callable (bind dataset-specific config, e.g. `cfg`/`n_context`, via
  `functools.partial` before passing it in) instead of knowing about SSVI or sampling itself. Each epoch builds
  batches via `build_regression_batches`, caches each one's context into the TabPFN executor with no gradient
  (`fit_from_preprocessed(..., no_refit=True)`), then does a differentiable forward pass on the query points
  to get bar-distribution logits and backprops a regression loss (`_compute_regression_loss`) against the
  model's own bar-distribution buckets. `run_name` is a required positional arg (no default) — checkpoints
  always save to `<repo_root>/checkpoints/<run_name>/`, resolved from `finetune.py`'s own file location so it's
  correct regardless of caller cwd (e.g. a notebook running with cwd=`notebooks/`). Only `best.pt` (lowest val
  loss so far, or train loss if `n_val_surfaces=0`) and `final.pt` are saved — no per-epoch checkpoints, since
  each is ~224MB and epoch-by-epoch snapshots are rarely needed.
- **TabPFN estimator construction** — use plain `TabPFNRegressor(...)`, not
  `TabPFNRegressor.create_default_for_version(version=ModelVersion.V3, ...)`; the plain constructor already
  resolves to the V3 checkpoint by default, so the explicit version selection is redundant. `finetune.py`
  builds it with `fit_mode="batched"`, `n_estimators=1`, and `inference_config={"FINGERPRINT_FEATURE": False}`.
  **`n_estimators=1` is required for training stability, not just a style choice**: bumping to 2 (to match
  the library's own `FinetunedTabPFNRegressor` default) mixes TabPFN's two different default preprocessing
  branches (squashing+SVD vs. quantile_uni) into one training run, and empirically caused the forward pass
  to emit `-inf` logits within the first epoch, permanently corrupting the model weights once
  `loss.backward()`/`optimizer.step()` ran on the resulting `inf` loss (loss stays `inf` for every epoch
  after). This is consistent with `notes/tabpfn_preprocessing_ablation.md`, which already found that
  ensembling both branches underperforms using the squashing+SVD branch alone at inference time — there's no
  evidence `n_estimators=2` is worth the instability. Root cause of the `-inf` forward pass itself is not
  fully diagnosed; if raising `n_estimators` again, re-verify training stability first (e.g. check
  `torch.isfinite(loss)` every batch before backprop).
- **`src/evaluation/surface_eval.py`** — `check_arbitrage` checks a predicted/generated surface for
  calendar-spread and butterfly arbitrage violations in total-variance space (`w = iv^2 * ttm`).
- **`notebooks/`** — exploratory work: `tabpfn_test.ipynb` (baseline, non-finetuned TabPFN on SSVI surfaces),
  `tabpfn_finetuning.ipynb` (drives/inspects the finetuning loop), `ssvi_validation.ipynb` (sanity-checks
  the SSVI generator/arbitrage conditions).
- **`notes/tabpfn_preprocessing_ablation.md`** — findings on which TabPFN preprocessing steps matter for
  this `(k, tau) -> IV` task: squashing-scaler and SVD features are load-bearing and must be kept; the
  fingerprint feature is dead weight for this dense, non-duplicated grid and is disabled
  (`FINGERPRINT_FEATURE: False` in `finetune.py`) — don't re-enable it without re-running the ablation.
- **`questions.md`** — open research questions being tracked (parameter priors, grid/context sizing,
  whether to give the model context vs. sparse quotes only). Check before assuming a design choice is final.

## Conventions

- Surfaces are always represented as `(n_ttm, n_k)` grids; flattening order is `ttm`-major (`TT, KK = meshgrid(ttms, ks, indexing="ij")`, then `.ravel()`).
- Context/sparse-point sampling is always Gaussian-weighted toward ATM in `k` and uniform in `ttm`; this logic lives solely in `data_preperation.py` (`sample_sparse_points`) — `finetune.py` is data-agnostic and has no sampling logic of its own.