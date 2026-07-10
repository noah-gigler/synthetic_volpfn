# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Masters semester project studying whether TabPFN (a tabular foundation model) can be finetuned to
interpolate/extrapolate implied volatility surfaces from a sparse set of quoted points, using synthetic
SSVI-generated surfaces as training data (see `VolSmoothing_with_TabPFN_proposal.pdf`).

## Code style

- No docstrings. No comments restating what the code obviously does.
- A `#` comment is only for the non-obvious: a formula/theorem reference, a units/shape
  note, a gotcha (e.g. `regime=0 -> bid=ask=true`). If it wouldn't confuse a reader,
  don't write it. Match `src/data_generation/SSVI.py`'s density, not `data_preperation.py`'s.

## Code style

- No docstrings. No comments restating what the code obviously does.
- A `#` comment is only for the non-obvious: a formula/theorem reference, a units/shape
  note, a gotcha (e.g. `regime=0 -> bid=ask=true`). If it wouldn't confuse a reader,
  don't write it. Match `src/data_generation/SSVI.py`'s density, not `data_preperation.py`'s.

## Commands

- Environment is managed with `uv` (Python 3.12 pinned via `.python-version`); dependencies in `pyproject.toml` / `uv.lock`.
- Run a module: `uv run python -m src.data_generation.data_preperation` or `uv run python -m src.model.finetune`
- No test suite, linter, or build step currently exists in this repo.
- Notebooks are stripped of output on commit via `nbstripout` (declared dependency).

## Architecture

Data flow: `config.yaml` (SSVI prior + grid settings) → `src/data_generation` → `src/model/finetune.py` → `checkpoints/`.

- **`config.yaml`** — single source of truth for the SSVI parameter prior (median/sigma per parameter,
  sampled as lognormal/logit-normal) and the `(z, ttm)` grid shape (standardized moneyness `z`, not raw
  `k`). Both data generation and finetuning load this file directly.
- **`src/data_generation/grid.py`** — `Grid(cfg)` is the single source of truth for the evaluation
  lattice: it computes the `(ttms, zs)` axes and the flattened ttm-major `tau/z/k` arrays (plus `rho=√τ`,
  `shape`) **once**, and exposes `features()` (the model's `[z, tau]` view). The `k = z·√τ` law lives in
  exactly one place — the `z_to_k`/`k_to_z` helpers here. Everything downstream builds a `Grid` and pulls
  the coordinate it needs instead of re-`meshgrid`-ing or writing `√τ` inline. **The model is fed `z`, not
  physical `k`** (the wedge self-similarity lives in `z`; feeding raw `k` empirically hurt the sparse-context
  regime — see `notes/results_summary.md`); physical `k` is used only where the physics demands it (BS
  spread pricing in `noise.py`, and the SSVI fit/eval, which are intrinsically `k`-parametrized).
- **`src/data_generation/SSVI.py`** — samples SSVI parameters from the prior and evaluates the SSVI
  parametrization on a `(ttm, k)` grid (physical `k`, from `Grid.k`), vectorized over a batch of surfaces.
  Enforces butterfly-arbitrage bounds on `eta` (Gatheral & Jacquier 2013, Theorem 4.2) when sampling.
- **`src/data_generation/data_preperation.py`** — turns full surfaces into (context, query) point sets:
  `sample_sparse_points` draws a Gaussian ATM-weighted (in `z`, `z=0` is ATM), uniform-in-ttm subset of grid
  indices as the "sparse quotes" context; `data_preparation` builds the corresponding train/test
  `(X=[z,tau], y=iv)` arrays per surface (via `Grid`). `generate_surfaces` returns `(grid, surfaces)`. `n_context` may be an int or a `(lo, hi)` tuple — `sample_context_sizes` then draws a
  per-surface size (uniform by default, `dist="log"` for log-uniform). `make_stratified_eval_set`
  presents the *same* surfaces at several fixed context sizes (size-major) for stable validation.
  Has a `__main__` smoke test that fits/predicts a single surface with vanilla `TabPFNRegressor`.
- **`src/data_generation/noise.py`** — bid-ask quote noise and the noisy data providers
  (`noisy_data_preparation`, `make_noisy_stratified_eval_set`), mirroring `data_preperation.py`'s
  contracts so they plug into `finetune()` unchanged. Layered spread model (vega-based structural
  half-spread from `tick + beta*price`, per-surface lognormal regime, per-quote lognormal jitter,
  cap), deliberately not recoverable from `(z, tau)` alone; the true IV sits at a uniform random
  position inside the quoted spread (asymmetric — no mid is ever observed). The spread model prices in
  physical `k = Grid.k`, but the feature schema is `X = [z, tau, side]` with side −1=bid/+1=ask/0=true; each quote location contributes two context
  rows, query rows are the full grid with side=0 and clean targets, and `n_context` counts quote
  *locations* (a context holds `2*n_context` rows — `finetune`'s val-breakdown labels show rows).
  Noise parameters live in `config.yaml` under `noise:`. Also hosts the truth-free providers for
  the quote loss: `quote_data_preparation` (context = bid/ask rows; query = full grid with
  `y_query` an `(n_grid, 2)` array of `[bid, ask]` interval targets at *held-out* quote locations,
  NaN elsewhere — true prices appear nowhere; the affine z-norm in `preprocess_surfaces` transforms
  interval bounds correctly with no special handling) and `make_quote_eval_set` (frozen proxy-loss
  val set, so checkpoint selection stays truth-free too).
- **`src/model/SSVI.py`** — least-squares SSVI refit baseline (`fit_ssvi`, prior-median +
  prior-sampled restarts, fit in total-variance space; optional per-point `weights` for noisy
  quotes: `1/(2*y*tau*s)` with half-spread `s`), plus `predict_ssvi`. On clean quotes this recovers
  surfaces exactly at ≥6 points (parameter count) — it is the oracle baseline there.
- **`src/model/preprocessed_dataset.py`** — `preprocess_surfaces(estimator, train, test, rng)` turns a
  `data_provider`'s pre-split `(context, query)` arrays into a list of TabPFN `RegressorBatch`es, one per
  surface, or a group of up to `group_size` equal-context surfaces stacked on the dataset-batch dim (naming
  note: "batch" here is the TabPFN dataset-batch dim, distinct from the `batch_size` gradient-accumulation
  notion in `finetune.py`; `group_size=1` gives one surface per `RegressorBatch`). Deliberately *not* using TabPFN's own `tabpfn.finetuning.data_util.get_preprocessed_dataset_chunks`:
  that helper assumes one raw dataset that still needs a generic `split_fn`-based train/test split (plus lazy
  chunking for oversized datasets), neither of which applies here since context/query are already split and
  small. Preprocessing-config selection (`_initialize_dataset_preprocessing`, i.e. which pipelines/target
  transforms TabPFN picks) is run on **context only**, matching TabPFN's own `fit(X_train, y_train)` semantics
  (config selection never sees test data in normal use) and how a real deployment would work (only sparse
  quotes are ever available, never a full grid) — this differs from the old finetuning-helper pattern, which
  selected configs based on train+test combined, and is a deliberate choice, not an oversight. Surfaces in one
  epoch can have different context sizes; nothing here assumes a fixed `n_context`.
- **`src/model/quote_loss.py`** — the truth-free training loss ("quote loss"): interval-censored
  NLL `-log P(bid ≤ y ≤ ask)` via the bar distribution's differentiable `cdf` at held-out quote rows
  (NaN rows masked — but filled with a dummy value first, `cdf` asserts on NaN), plus calendar and
  butterfly hinge penalties on the raw-space pointwise mean over the full grid (Durrleman-g, same
  formulas as `check_arbitrage`). Assumes query = full ttm-major grid. Plugs into `finetune()` via
  the `loss_fn` hook; bind `grid_shape`/`lambda_cal`/`lambda_bf` with `functools.partial`. Headline
  result (see `notes/results_summary.md`): at N=20 quotes this matches the fully supervised model's
  truth-MAE without ever training on a true price, with 0% arb violations.
- **`src/model/finetune.py`** — the actual meta-learning finetuning loop, data-agnostic: it takes a
  `data_provider(n) -> (train, test)` callable (bind dataset-specific config, e.g. `cfg`/`n_context`, via
  `functools.partial` before passing it in) instead of knowing about SSVI or sampling itself. Each epoch
  preprocesses fresh surfaces via `preprocess_surfaces`, caches each one's context into the TabPFN executor
  with no gradient (`fit_from_preprocessed(..., no_refit=True)`), then does a differentiable forward pass on
  the query points to get bar-distribution logits and backprops a regression loss (`_compute_regression_loss`)
  against the model's own bar-distribution buckets. `batch_size` is an *effective* batch size implemented as
  gradient accumulation (gradients averaged per optimizer step). Surfaces with equal context sizes *are*
  run as a single true batched forward pass via `group_size`: `preprocess_surfaces(..., group_size=G)` packs
  up to `G` consecutive equal-context surfaces onto TabPFN's dataset-batch dim, and `_run_pass` unpacks the
  resulting `B` dimension (same path eval uses at `group_size=16`). To use it, have the `data_provider` draw
  equal context sizes per group (`data_preparation(..., size_group=G)`) and pass `group_size=G` with
  `batch_size` a multiple of `G` (asserted). Ragged context sizes only prevent stacking *across* different
  sizes, not batching within a size; `group_size=1` reproduces the old one-surface-per-forward path
  byte-identically. Validation: pass a prebuilt
  `val_data=(train, test)` (e.g. from a stratified eval-set helper) and `val_every=k`; val surfaces are
  preprocessed **once** up front (rebuilding would re-consume RNG and drift the val task), the log prints a
  per-context-size loss breakdown (labels = context *rows*), and `best.pt` only updates on val epochs. With
  fresh surfaces each epoch there is no dataset to overfit — prefer `final.pt` (gets the full cosine anneal);
  `best.pt` selection is dominated by the smallest-context losses. An optional
  `loss_fn(estimator, surface_batch, logits_BQL)` replaces the default CRPS+MSE loss for both train
  and val passes (used by the quote loss); `loss_fn=None` keeps the default path byte-identical. `run_name` is a required positional arg (no default) — checkpoints
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
  calendar-spread and butterfly arbitrage violations in total-variance space (`w = iv^2 * ttm`);
  `eval_surfaces` (MAE/MAPE/arb rates; schema-agnostic, uses the fit-then-`reload_state` pattern for
  finetuned checkpoints), `quantile_coverage` (empirical coverage of central predictive intervals via
  TabPFN quantile output) and `inside_spread_fraction` (predictions within [bid, ask] at quote locations;
  note the synthetic truth *always* lies inside the spread by construction, so ~100% is expected of a good
  model — the metric only flags pathologies, it cannot distinguish denoising from mid-interpolation).
- **Evaluate in a fresh kernel/process — known stale-kernel artifact.** Long-lived Jupyter kernels have
  twice produced corrupted eval results at specific (model, context-size) slots: 10-50x worse MAE,
  deterministic within the session (same slots across independent data draws), different slots each
  session, affecting even the vanilla non-finetuned baseline, and never reproducible in a fresh process.
  A controlled ordering experiment (same slot evaluated before vs after a sweep-like history in one fresh
  process) showed no effect, ruling out a simple shape-keyed cache; root cause unknown. Mitigation:
  restart the kernel before eval cells, or run sweep slots in subprocesses; distrust any in-kernel number
  that contradicts training-val values. The finetune-style path (`fit_from_preprocessed` + `forward`) has
  been immune throughout.
- **`notebooks/`** — exploratory work: `tabpfn_test.ipynb` (baseline, non-finetuned TabPFN on SSVI surfaces),
  `tabpfn_clean_finetuning.ipynb` (drives/inspects the clean finetuning loop), `tabpfn_supervised_finetuning.ipynb`
  (bid/ask-noise experiment, supervised on true IV: run cell + sweep/coverage/inside-spread/visual eval),
  `tabpfn_arb_finetuning.ipynb` (quote/arb-loss experiment — truth-free training; offline truth eval is
  reporting only, never checkpoint selection), `ssvi_validation.ipynb` (sanity-checks the SSVI
  generator/arbitrage conditions).
- **`notes/results_summary.md`** — running summary of all finetuning experiments and established
  findings (run-by-run outcomes, refit identifiability, the noisy-quote headline results, calibration,
  known issues, open items). Read this before designing a new run or re-deriving conclusions.
- **`notes/tabpfn_preprocessing_ablation.md`** — findings on which TabPFN preprocessing steps matter for
  this `(k, tau) -> IV` task: squashing-scaler and SVD features are load-bearing and must be kept; the
  fingerprint feature is dead weight for this dense, non-duplicated grid and is disabled
  (`FINGERPRINT_FEATURE: False` in `finetune.py`) — don't re-enable it without re-running the ablation.
- **`questions.md`** — open research questions being tracked (parameter priors, grid/context sizing,
  whether to give the model context vs. sparse quotes only). Check before assuming a design choice is final.

## Conventions

- Surfaces are always represented as `(n_ttm, n_z)` grids; flattening order is `ttm`-major (`meshgrid(ttms, zs, indexing="ij")` then `.ravel()`, all done once inside `Grid`). Build a `Grid(cfg)` and read `g.z`/`g.k`/`g.tau`/`g.shape` — don't re-`meshgrid` or write `√τ` inline; the `k = z·√τ` law lives only in `grid.py` (`z_to_k`/`k_to_z`).
- **The model's moneyness feature is standardized `z`, not physical `k`.** `k` appears only where the physics needs it: BS spread pricing in `noise.py` and the SSVI fit/eval (`fit_ssvi`, `predict_ssvi`, `generate_surfaces`). Any `z↔k` boundary conversion goes through `z_to_k`/`k_to_z`.
- Context/sparse-point sampling is always Gaussian-weighted toward ATM in `z` (`z=0` is ATM) and uniform in `ttm`; this logic lives solely in `data_preperation.py` (`sample_sparse_points`) — `finetune.py` is data-agnostic and has no sampling logic of its own.