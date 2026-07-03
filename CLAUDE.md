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
- **`src/model/finetune.py`** — the actual meta-learning finetuning loop. Each gradient step processes one
  synthetic surface: sample it, split into context/query (same ATM-weighted scheme as above), run TabPFN's
  own preprocessing pipeline (`get_preprocessed_dataset_chunks`), cache the context into the TabPFN executor
  with no gradient (`fit_from_preprocessed(..., no_refit=True)`), then do a differentiable forward pass on
  the query points to get bar-distribution logits and backprop a regression loss (`_compute_regression_loss`)
  against the model's own bar-distribution buckets. Saves a checkpoint every epoch to `output_dir`
  (default `checkpoints/finetune_v1/`). Currently SSVI-specific (takes `cfg` directly and builds surfaces
  inline) — a planned refactor moves data generation/sampling behind a generic
  `data_provider(n) -> (train, test)` callable so the loop works on other synthetic generators or real data
  without change; not yet implemented.
- **TabPFN estimator construction** — use plain `TabPFNRegressor(...)`, not
  `TabPFNRegressor.create_default_for_version(version=ModelVersion.V3, ...)`; the plain constructor already
  resolves to the V3 checkpoint by default, so the explicit version selection is redundant. `finetune.py`
  builds it with `fit_mode="batched"`, `n_estimators=2` (matches the library's own
  `FinetunedTabPFNRegressor` default for the training phase; bump to 8 for final inference after
  finetuning), and `inference_config={"FINGERPRINT_FEATURE": False}`.
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
- Context/sparse-point sampling is always Gaussian-weighted toward ATM in `k` and uniform in `ttm` — this logic is duplicated in `data_preperation.py` and `finetune.py`; keep them consistent if changing the sampling scheme.