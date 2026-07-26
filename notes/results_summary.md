# Results summary (as of 2026-07-26)

Everything before the `z`-reparametrization + widened grid (`config.yaml`: `z∈[-1.5,0.5]`,
`ttm_min=0.02`, commit `1c4c12e7` onward) is archived in git history, not repeated here — those
numbers are on a materially different (narrower, `k`-parametrized) task and aren't a fair
comparison point anymore. This doc covers the current grid/config only.

## Current best model

**`ssvi_supervised_gs4_12h`** — supervised (true-IV) loss, `group_size=4, batch_size=16`,
2400 epochs × 30 steps/epoch = **72,000 optimizer steps** (~12h on an RTX Pro 6000 Blackwell).
Checkpoint + logs in `checkpoints/ssvi_supervised_gs4_12h/`. A `ssvi_supervised_gs4_24h` run
(same recipe, 4800 epochs = 144,000 steps) is in progress to see if the trend continues past 72k;
not yet evaluated.

Beats every other checkpoint we have (`ssvi_supervised_gs1_15k`, `ssvi_supervised_gs4_15k`, and an
externally-trained `ssvi_supervised_overnight_lightning`, 16k steps/batch_size=32) on identical
evaluation draws, at every tested `n_ctx ∈ {3,5,10,20,40,60}` and every `m ∈ {0.5,1,2}`. At `m=1.0`
it matches or beats the SSVI refit oracle at *every* context size tested, including `n=60` (FT
0.0017 vs SSVI 0.0021) — normally SSVI's strongest regime. At `m=0.5`/`m=2.0`, SSVI still pulls
ahead from `n≈20` on, as expected.

## Established findings

1. **`group_size` batching trades gradient-update frequency for GPU throughput — this is the
   single biggest lever, bigger than any noise/loss-design question.** `batch_size` is the
   gradient-accumulation window; `total_steps = n_epochs · ⌈n_surfaces_per_epoch / batch_size⌉`.
   An earlier run at `group_size=128, batch_size=256` got only ~1000 optimizer steps despite
   processing hundreds of thousands of surfaces — catastrophically undertrained (lost to SSVI
   almost everywhere it used to win). `group_size=1` (no batching) reproduces old per-surface
   behavior exactly but is slow (~206ms/surface on current hardware). Benchmarked sweep
   (supervised, `n_context=(3,60)`):

   | group_size | ms/surface | 15k-step wall-time estimate |
   |---|---|---|
   | 1 | 206.5 | 3.44h |
   | 2 | 109.0 | 3.63h |
   | 4 | 60.4 | 4.02h |
   | 8 | 38.9 | 5.18h |
   | 16 | 25.0 | 6.68h |
   | 32 | 19.9 | 10.60h |
   | 64 | 18.6 | 19.81h |

   `group_size=4` is the practical sweet spot: near-`group_size=1` gradient diversity (still 4
   distinct context-size groups per accumulation step at `batch_size=16`) while getting most of
   the throughput win. `group_size=4/batch_size=16` beat `group_size=1/batch_size=4` at matched
   step count in every test.
2. **Step count, not epoch count or wall-clock, is what matters — and more steps continues to
   help well past 15k.** `gs4_15k` (15k steps) → `gs4_12h` (72k steps) closed most of the gap to
   history at `m=1.0` (e.g. `n=40`: gap from 76% to ~8%) and pushed FT to beating SSVI outright at
   several slots. Diminishing but still real returns past 15k steps.
3. **Regime (noise level `m`) doesn't uniformly favor FT at higher noise — it's context-size
   dependent.** FT/SSVI MAE ratio (lower = better for FT), `gs4_12h`:

   | n_ctx | m=0.5 | m=1.0 | m=2.0 |
   |---|---|---|---|
   | 3 | 0.894 | 0.817 | 0.590 |
   | 5 | 0.606 | 0.507 | 0.496 |
   | 20 | 1.211 | 1.028 | 1.341 |
   | 40 | 1.412 | 1.045 | 1.154 |
   | 60 | 1.143 | 0.810 | 1.333 |

   At low context, higher noise *does* help FT relatively (weaker likelihood → learned prior
   worth more, as expected). At high context it inverts — SSVI is the statistically efficient
   estimator once it has enough correctly-weighted points, and that advantage compounds fastest
   in the highest-noise regime. `m=1.0` looks like the "best" regime only because it's the
   context-averaged sweet spot between these two opposing effects, not because it's special.
4. **Noise correlation (`rho`) mismatch, not correlation itself, is the real risk.** Added a
   single-factor Gaussian-copula `rho` parameter to `noise.py` (`rho=0` → i.i.d. per-quote noise,
   `rho=1` → one shared draw per surface). Models trained and evaluated on *matched* rho perform
   comparably regardless of rho value. Cross-tested (train rho=0, test rho=1 or vice versa)
   degrades sharply at high context (e.g. `m=1.0, n=60`: matched-rho MAE ~0.008-0.009 either way;
   mismatched climbs to ~0.02, roughly 2.5x worse) — the model specializes to whatever noise
   structure it saw in training and doesn't generalize across the mismatch.
5. **SSVI refit is underdetermined below `n_ctx≈6`** (6 free parameters) — fits reachieve exactly
   zero training residual on any random n=3 draw regardless of the true surface, but out-of-sample
   MAE varies 5-6x across refits of the *same* data (measured: 0.0085 to 0.0522 MAE across 15
   identical-data refits). Any SSVI number at `n_ctx<6` should be read as high-variance noise, not
   a stable comparison point.
6. **`regime=0` is now genuinely noiseless.** Previously `regime=0` only zeroed the price-proportional
   half-spread term but left jitter and random in-spread placement active — not actually clean.
   Fixed in `add_quote_noise` (`noise.py`) with a hard early-return (`bid=ask=true`) rather than
   scaling, since `tick`'s presence in `half_spread` must stay untouched for `regime>0`.

## Session 2 (2026-07-17) — pipeline overhaul, a real training bug found+fixed, several ablations

**Everything in the "Arbitrage (OpDS quote-loss) investigation" section below predates a
significant bug fix (finding 13) and should be read as historical context for how the
investigation unfolded, not as current numbers.** The new findings in this section supersede it.

13. **Side-column label leak (major bug, fixed).** `sample_arb_grid` (`grid.py`) used to mark each
    arb-grid row's type via literal sentinel values on the `side` feature column
    (`BUTTERFLY=2.0`/`CAL_LOW=3.0`/`CAL_HIGH=4.0`, only meant for `quote_loss.py`'s row-type
    bookkeeping) — but those same rows were also fed to the model as `X_query` input, meaning the
    model saw a literal "this is a butterfly-check point" signal on every arb-grid query it ever
    trained on. This let it partially shortcut the arb penalty via the marker rather than learning
    genuine (z,tau)-only smoothness. Fixed: markers removed entirely; `side` is `0.0` on every
    arb-grid row (matching a real query), and row-type bookkeeping in `quote_loss.py` is now done
    by pure position/count arithmetic (`arb_grid_shape(cfg)` gives the fixed axis counts needed to
    recover row boundaries without any marker). Retraining
    `curvature_jitter` under the fix (`ssvi_opds_arb_curvature_jitter_v2`, exact original
    hyperparameters) shows a large, real improvement at the same eval slot (regime=1, n_ctx=20):
    cell_frac **10.74%→1.80%** (~6x), arb_free% **3.3%→14.5%** (~4x) — confirming the leak had
    real cost, not just a theoretical concern.
14. **The `group_size=4`/long-step-count recipe (finding 1-2) generalizes to arb loss too**, closing
    the open item from session 1. `ssvi_opds_arb_curvature_jitter_gs4_15k` (group_size=4,
    batch_size=16, 15k steps, same loss as v2) beats `v2` (group_size=20, batch_size=40, 6k steps)
    on both axes at regime=1, n_ctx=60: MAE 0.0069→0.0053 (23% better), arb_free% 10.4%→17.4% (67%
    relative better). A full-budget (72k-step) arb run under the *fixed* NLL (finding 17) is the
    natural next step — an earlier attempt at this under the *pre-fix* loss was cancelled and its
    checkpoint dir deleted once the NLL bug was found, to avoid wasting the 72k-step budget on
    stale loss math.
15. **Eval pipeline overhaul**: `EVAL_N` raised 50→512 surfaces/slot (24 slots = 12,288 surfaces
    total), `eval_surfaces` (MAE pass) and `eval_arbitrage_fine` (arb pass) both batched via
    `preprocess_surfaces`/`group_size` instead of one-surface-at-a-time (`eval_surfaces` default
    `group_size=128`, cheap: ~11ms/surface; `eval_arbitrage_fine` default `group_size=8`, the arb
    grid is ~10x bigger per surface so it OOMs above that — was previously defaulted to 16, which
    now OOMs on the finer eval-only grid, fixed). The SSVI baseline refit
    (`load_baselines`/`_baselines`) is now parallelized via `ProcessPoolExecutor` (embarrassingly
    parallel per-surface `scipy` fits, CPU-only, no GPU needed) — serial cost for N=512 was
    projected at ~18min, parallel across 32-64 cores brings it to ~1.5min, and it can run on a
    completely separate CPU-only allocation from the GPU eval passes (`--baseline-jobs`).
    `eval_arbitrage_fine` now takes a *frozen* arb grid (`load_arb_grid`, seeded once, reused
    across every checkpoint) instead of resampling a fresh random grid per surface — training's own
    `sample_arb_grid` randomizes the tau-row step every call, which is correct for training
    (fresh grid every step) but meant no two eval calls used the same grid, blocking both batching
    (unequal query shapes can't stack) and apples-to-apples comparison across checkpoints/runs.
    Val sets are also stratified: 128 surfaces per context size, split evenly across
    `EVAL_REGIMES=[0,0.5,1,2]` (was a single continuous `lognormal` draw, unstratified, only
    20-ish surfaces per context size) — `val_group_size` (128 supervised, 8 arb, independent of
    training's own `group_size`) controls val-pass batching.
16. **Noise-set size-planning (established via direct measurement, not guessing)**: relative
    standard error at N=50 (the old default) was ~11-13% for MAE and worse for arb metrics —
    `arb_free%` (a low-mean Bernoulli, ~17-19% base rate) is the noisiest metric by far, ~29-31%
    relative SE at N=50, still ~9-10% at N=512; `worst_cell` is an order statistic (min), for which
    `std/√N` isn't a rigorous SE, treat it as directional only. `cell_frac`/`mean_depth`/MAE behave
    like clean i.i.d. means (bootstrap SE matches the analytic `std/√N` almost exactly) and reach
    single-digit-% relative error by N≈256-512. This is why `EVAL_N=512` was chosen (finding 15).
17. **A second, more fundamental arb-loss bug (found via root-causing the negative-IV outlier
    below), fixed.** `quote_arb_loss`'s interval NLL is `-log(CDF(ask) - CDF(bid))`, correct for a
    genuine nonzero-width interval — but at `regime=0` (`add_quote_noise` returns `bid=ask=true`
    exactly), the interval has *zero width*, and for any continuous distribution
    `CDF(x) - CDF(x) = 0` **by construction**, regardless of whether the prediction is good or bad.
    That NLL term silently floors to the `min_prob` clamp's constant value every time, carrying
    zero gradient toward the true value — at `regime=0` the model was only ever getting shape
    (`cal`/`bf`) pressure, never an accuracy signal. Fixed: zero-width rows now use the point/bucket
    cross-entropy (`bardist.forward`, the same mechanism the plain supervised path already uses)
    instead of the interval CDF formula; genuine (nonzero-width) intervals are unaffected. Not yet
    retrained under this fix at the time of writing (`ssvi_opds_arb_curvature_jitter_gs4_15k_v2` is
    in progress) — expected to reduce or eliminate finding 18's negative-IV pathology, since it's
    the direct mechanism that made `regime=0` an under-constrained blind spot.
18. **Negative-IV prediction outlier, characterized and root-caused (pre-finding-17-fix).** Arb-loss
    checkpoints occasionally predict a **negative** raw IV at isolated grid points (regime=0,
    n_ctx=40: up to 20% of surfaces have ≥1 negative point). Confirmed **not** a resolution/
    extrapolation artifact — retested on the *training*-resolution grid (coarser than eval's) and
    the problem is if anything worse there (`worst_cell` as low as -1.6M vs -8.1K on the finer eval
    grid), so it's a genuine training-time model behavior, not an eval-grid-mismatch artifact.
    Strictly confined to `regime=0`: exactly 0.0% of surfaces show any negative prediction at
    regimes 0.5/1/2, across every context size tested — directly explained by finding 17 (regime=0
    is the one setting where the NLL provides no accuracy gradient at all). A negative prediction
    gets clamped to `1e-3` before the arb metric's `1/w` (total-variance) terms, creating an
    artificial near-zero dip between otherwise-normal neighboring predictions, which the metric's
    finite-difference second derivative amplifies into an enormous (theoretically unbounded)
    apparent violation depth — this is why `mean_depth`/`worst_cell` (plain `.mean()`/`.min()`
    across surfaces) get dominated by a handful of outlier surfaces: at regime=0, n_ctx=40,
    `mean_depth` was -24.12 by mean vs -1.38 by median (17x), `worst_cell` -566,218 vs -7.97
    (71,000x) — while regimes 0.5/1/2 (zero negative predictions) show mean and median close
    together, confirming the distortion is entirely a regime=0 artifact. Decision: retrain under
    the finding-17 fix first rather than patch the reporting (median-based `eval_arbitrage_fine`
    reporting) — if the fix resolves the root cause, the metric distortion goes away on its own.
19. **rho ablation — training-time noise correlation, matched-eval comparison.** `rho` (single-
    factor Gaussian copula, `rho=0`=iid per-quote noise, `rho=1`=one shared draw per surface,
    added in session 1 finding 4) re-tested with a *matched* eval set this time (same rho used for
    training and eval, not the mismatched cross-test from session 1). `rho=1` (`gs4_15k` recipe)
    dramatically **beats** the `rho=0` baseline on MAE at larger context, and the gap *grows* with
    context rather than shrinking. Regime m=1, FT MAE (rho=0 vs rho=1):
    - n_ctx=3: 0.0296 (rho=0) vs 0.0280 (rho=1) — small gap
    - n_ctx=20: 0.0055 vs 0.0035 — rho=1 ~1.6x better
    - n_ctx=60: 0.0029 vs 0.0016 — rho=1 ~1.8x better, and **beats the SSVI refit oracle**
      (0.0045) by ~2.8x, a regime where SSVI normally wins or ties
    Mechanism: at `rho=1`, `_corr_normal`'s per-quote term (`sqrt(1-rho)·z_quote`) vanishes
    entirely (`sqrt(1-1)=0`) — the whole surface's noise reduces to just 2 shared scalars (one for
    spread-width jitter, one for in-spread placement), not N independent draws. With enough
    context the model can identify those 2 scalars and algebraically invert back to the true IV
    almost exactly, which is why MAE keeps *shrinking* with more context instead of plateauing at
    a noise floor the way it does under genuine per-quote noise. SSVI's weighted least-squares
    refit doesn't exploit this collapsed structure (treats each residual as independent), so it
    loses its usual large-context advantage here. Arb-freeness is a wash between rho=0/rho=1
    (`cell_frac` averaged across context sizes per regime: within 0.1-0.4pp of each other, no
    consistent winner) — the rho effect is specifically an MAE/noise-model story, unrelated to
    arbitrage structure. A `rho=0.5` run (same recipe) is queued/in progress to fill in the
    middle of this curve; not yet evaluated.
20. **`FEATURE_SHIFT_METHOD` ablation.** TabPFN's ensemble feature-position shift (`"shuffle"` by
    default, shuffles feature column order per ensemble member to emulate position-invariance)
    matters even at `n_estimators=1`: disabling it (`None`) hurts MAE consistently across every
    context size tested, worse at small context (regime m=1: n_ctx=3 0.0296→0.0410, +38%; n_ctx=60
    0.0029→00033, +14%). Keep the default; no reason found to disable it. (`"rotate"`, the third
    option, not yet tested.)
21. **wandb integration added** (`wandb.ai/volpfn/volpfn`) — per-epoch scalar logging built into
    `finetune()`, on by default via `run_finetuning.py`'s CLI. See `CLAUDE.md`'s "Weights & Biases"
    section for the compute-node file-upload caveat (files need a separate login-node step).

## Arbitrage (OpDS quote-loss) investigation — open, not resolved

7. **The eval-side arb metric was misleading.** `check_arbitrage`/`eval_surfaces` report
   `.any()` per surface (fraction of surfaces with ≥1 violating cell somewhere on the *coarse*
   25x15 grid), not fraction of the domain that's actually violating. Numbers like "50-100% arb"
   quoted mid-investigation were this `.any()` statistic on a 6,000-cell fine grid — with a true
   cell-level violation rate of ~1%, nearly every surface will have "at least one" violating cell
   by chance, making the metric look catastrophic when it isn't. **Always check true cell-level
   fraction + depth on a fine grid, not just the `.any()` numbers, before concluding a model is
   broken.**
8. **True fine-grid (4x resolution) violation rate is small and localized, not universal.**
   `ssvi_opds_arb_long_lamda_10` (`lambda_cal=lambda_bf=10`, up from the original `1.0`):
   cell-level butterfly violation ~0.9%, concentrated almost entirely in a specific moneyness band
   `z≈-0.9 to -0.5` (3-8% local rate) with `z>-0.3` completely clean (0%). Calendar arb is
   essentially solved (~0%) throughout — its training grid (`sample_arb_grid`'s calendar branch)
   is dense and deterministic in tau (~43 fixed rows, no randomization), unlike butterfly's sparse
   random rows, and that density difference tracks the outcome closely.
9. **Butterfly's training-time tau sampling has a real, confirmed bug**: `r_b =
   np.arange(rho_lim[0], rho_lim[1], np.random.uniform(0.075, 0.125))` randomizes only the *step*,
   never the *phase* — every draw anchors at the exact same `rho_lim[0]`, and `rho_lim[1]`
   (longest maturity) is never included at all (`arange` excludes the stop bound). Not yet fixed.
10. **Violators-only loss reduction (mean over violating cells only, not the whole grid) made
    things worse, not better, when tested.** Cell violation rate went 0.9%→1.2%, mean depth
    -0.14→-0.23, worst cell -0.64→-0.96, and the violation footprint *spread* into previously-clean
    regions (z as far as -1.48). Likely explanation: violators-only removes the gradient dilution
    that was making corrections weak, but applied on top of the still-sparse/phase-anchored tau
    grid, the stronger per-point correction destabilizes the surface elsewhere between training
    steps (a worse "whack-a-mole" dynamic) rather than fixing coverage. **This contradicts the
    naive read of "violators-only always helps" — it needs the grid-density fix (finding 9) to
    actually pay off, matching what the historical real-data run history already implied
    (violators-only + coarse grid still left ~50% `.any()`-arb; only violators-only + fine+jittered
    grid together got to ~5%).**
11. **Comparison caveat: supervised (non-arb) models are never trained on off-grid query points at
    all** — their query is always the fixed native 25x15 grid, every epoch. Arb-loss models get
    (sparse, phase-biased) off-grid exposure via `sample_arb_grid`. So the earlier read that "arb
    loss only modestly beats plain supervised on fine-grid violation rate" (supervised: 50% of
    surfaces / 1.5-2.1% cells vs arb: 0.9-1.2% cells) is **not a clean comparison** — the arb
    models are being tested on exactly the kind of location they were partially trained for, the
    supervised ones never saw such points at all. An ablation (supervised loss + off-grid query
    exposure, no arb penalty) would be needed to isolate the two effects. Not yet run.

## Session 3 (2026-07-26) — Heston refit baseline, and the first cross-family evaluation

22. **The "SSVI refit beats the Heston refit on Heston data" result was a bug, not a finding.**
    The original Heston refit used QuantLib's built-in `HestonModel.calibrate`; it worked on clean
    quotes and fell apart on noisy ones, producing the alarming inversion. A hand-built fitter
    (`src/model/heston.py`, mirroring `src/model/SSVI.py`'s structure) beats the SSVI refit at
    *every* noise level and context size. Three things the built-in setup got wrong, all fixed:
    (a) fit in **total-variance** space (`w = iv²·τ`) with the same `1/(2·y·τ·s)` quote-noise
    weights the SSVI refit already used, not raw prices; (b) fit in **unconstrained coordinates**
    (log for the four positive params, logit for `-ρ`) rather than a box — unscaled, `kappa~2` and
    `v0~0.04` differ by ~50x and `trf`'s finite-difference Jacobian degrades; (c) price the
    **same contract the generator did**, via the shared `iv_at` in `data_generation/heston.py`
    (including its day-rounded τ) — the previous short-end mismatch was its own error source.
23. **Heston refit is an exact oracle on clean Heston quotes**, the direct analogue of SSVI's
    ≥6-point exact recovery on SSVI data: MAE **0.0000%** from `n_ctx≥10` at regime 0 over 512
    surfaces/slot. This is the correctness check that makes the noisy numbers trustworthy.
24. **The noisy Heston fit is genuinely ill-posed, not optimizer failure.** Diagnostic
    (`heston_data_cost`, data term only, so fitted and true parameter vectors are comparable): at
    *every* noisy setting, **100%** of fits reached an objective *below* the true-parameter
    objective, median ratio 0.84–0.87. Noise moves the minimum away from truth; the optimizer finds
    it correctly. This is the "narrow valley with a flat bottom" of Cui, del Baño Rollin & Germano
    (arXiv 1511.08718) showing up directly in our own numbers. Quantified: at regime 1 / n=10,
    `kappa` is off by **57%** (regime 4 / n=10: 72%) while surface MAE is only 0.61% — the
    degeneracy is in parameter space and barely touches the surface.
25. **Prior (Tikhonov/MAP) regularization helps exactly where theory says it should.** Penalizing
    toward `config.yaml`'s generating prior makes the fit a MAP estimate under the true prior. It
    is a large win at small context (regime 1, n=3: **1.79% vs 2.78%**) and neutral-to-marginally-
    worse at n≥40 — the textbook bias-variance crossover. Note `lambda_prior` is only in MAP units
    when `weights` are passed; against unweighted residuals (~1e-3 total-variance units) the
    standardized prior rows (~1) outweigh the data by ~10⁶ and the fit never leaves the prior
    median. Default is therefore `lambda_prior=0.0` — callers opt in.
26. **Anchoring `v0` is a negative result.** Fixing `v0` at the shortest-maturity ATM total variance
    (the standard literature recommendation for breaking the `v0`↔`kappa` degeneracy) is
    *consistently worse* than leaving it free. Empirically `v0` is the **best**-identified of the
    five parameters here (0.4–4% error), so anchoring spends a degree of freedom to pin down the
    one thing the data already determines. `anchor_v0=True` is still the signature default but no
    caller uses it.
27. **First cross-family eval: `ssvi_supervised_gs4_15k_heston_full` on a pure-Heston eval set**
    (`scripts/run_heston_eval.py`, `mixture.heston_frac=1.0`, 512 surfaces/slot, regime × n_ctx
    breakdown, both refits as baselines). MAE in %, regime 1:

    | n_ctx | FT | Heston | HestonMAP | SSVI |
    |---|---|---|---|---|
    | 3 | **1.34** | 2.78 | 1.79 | 4.57 |
    | 5 | **0.79** | 1.76 | 0.82 | 4.01 |
    | 10 | 0.44 | 0.59 | **0.42** | 0.98 |
    | 20 | 0.27 | 0.32 | **0.26** | 0.63 |
    | 40 | **0.16** | 0.19 | 0.17 | 0.52 |
    | 60 | **0.14** | 0.16 | **0.14** | 0.49 |

    Three things fall out of it:
    - **SSVI has a hard misspecification floor at ~0.5%.** It plateaus at 0.49–0.58% and never
      improves — not with more context (n=3→60 barely moves it past n=10), not with less noise
      (regime 2→0 changes it by 0.007%). Both Heston refit and the finetuned model keep improving;
      SSVI cannot. This is the cleanest structural result in the table and the empirical content of
      the proposal's "prior generalizes across families" claim.
    - **The finetuned model wins where the inverse problem is under-determined.** At n=3–5 under
      noise it beats the plain Heston refit by ~2.2x and beats even the MAP version. Five free
      Heston parameters against 3–5 quotes is under-determined per-surface; the amortized model is
      not. This is the empirical version of the amortization argument.
    - **At large context it ties the correctly-specified oracle** (n=60, regime 1: FT 0.14%,
      HestonMAP 0.14%, Heston 0.16%) while staying ~3.5x better than the misspecified one.

    Caveat worth carrying into the report: the finetuned model is *not* at its best on the clean
    set (regime 0, n=60: 0.19% vs 0.14–0.16% at noisy regimes). `regime=0` sets `bid=ask=true`,
    which is a different input distribution from anything it trained on (regimes are
    lognormal-sampled), so regime 0 is mildly out-of-distribution for it. Don't read regime 0 as
    "the easiest case" for the FT column.
28. **Arbitrage/UQ on the same run.** 80–95% of surfaces arb-free (worst at n=3), violated-cell
    fraction 0.00–0.06%. CRPS 0.0018–0.0103, 90% interval width 0.021–0.050, both tightening
    monotonically with context. This is a *supervised* checkpoint with no arb penalty in the loss,
    so the violation rate is expected and is **not** comparable to the quote/arb-loss runs.
29. **Rejected alternatives to QuantLib for Heston pricing** (benchmarked, scratchpad only).
    `pyfeng`: undeclared `statsmodels` dependency, FFT slower than QuantLib, 2.5e-2 relative price
    error at the shortest maturity. `stochvolmodels`: 42x slower with 2.2% median IV disagreement,
    traced to `vol_scaler = min(0.3, sqrt(v0*ttms[0]))` being sized off `ttms[0]` alone and tuned
    for ~100% crypto vol. `py_vollib_vectorized`: broken on modern numba, unmaintained since 2021.
    QuantLib at integration order 64 is exact to machine precision here. The real speed levers are
    `ProcessPoolExecutor` (4.4x on 8 cores, bit-identical) and QuantLib object reuse (1.33x), both
    now used.

## Bugs found and fixed this session

- **`eval_uncertainty` crashed on Heston data, `eval_surfaces` silently returned NaN.** Heston
  surfaces carry NaN in the deep-OTM/shortest-maturity corner (~0.2% of cells, float64 price
  underflow); SSVI surfaces have none, so neither path had ever been exercised. `eval_uncertainty`
  hit `bardist.cdf`'s NaN assert outright; `eval_surfaces` would have propagated NaN into the mean
  and reported a NaN MAE. Both fixed in `surface_eval.py` (fill-then-mask, the same order
  `quote_loss.py` uses). Note the first smoke run passed only because 8 surfaces didn't hit the
  corner — it took the 512-surface run to surface it.
- **Unpriceable-proposal residual scored as zero** (caught before it mattered). `_model_resid`'s
  first draft used `np.nan_to_num(r, nan=0.0)`, which would have scored a parameter vector that
  QuantLib cannot price *anywhere* as zero cost — a perfect fit, and a latent trap that would have
  quietly corrupted every noisy Heston result. Now charges the full observed variance instead.
- **`RuntimeError: stdDev (nan) must be non-negative`** from QuantLib `NPV()` when `trf` proposes a
  divergent parameter region. That's a verdict on the proposal, not an error — wrapped in
  try/except returning NaN at both the engine-construction and per-point levels.
- **Prior penalty ~10⁶ too strong** — see finding 25.

- **`preprocessed_dataset.py` `_stackable`**: only checked `X_context` shape equality before
  batching consecutive surfaces together, never `X_query`. Since `quote_data_preparation` draws a
  fresh random arb-grid (different query length) every `size_group` chunk, and context sizes are
  drawn from a wide range, two adjacent chunks occasionally draw the *same* context size by
  coincidence (~1/58 chance per pair, ~12,000 chunks in a 500-epoch quote-loss run → likely at
  least once) — silently merging surfaces with mismatched query shapes and crashing the `stack`
  call mid-training. Latent in both earlier arb runs; they just got lucky. Fixed: `_stackable` now
  also requires `X_query` shapes to match.
- **SSVI refit fed `z` where it expected physical `k`** in ad-hoc eval/notebook code — `fit_ssvi`/
  `predict_ssvi` need `k = z·√τ` (`z_to_k` in `grid.py`), not `z` directly. Silently wrong
  everywhere `τ≠1`. Fixed in `eval_run.py`'s SSVI comparison helper.

## Cluster/infra notes (Euler)

- `sbatch` needs **both** `--gpus=<type>:<count>` **and** an explicit `--partition=` pin (e.g.
  `cuda13pr.24h` for the RTX Pro 6000 pool). Omitting the partition silently let Slurm reroute jobs
  onto wrong-GPU-type nodes (40GB/80GB cards instead of the intended 96GB) twice, causing OOM
  crashes on `n_heldout=315`-scale quote-loss runs. `--gres=gpumem:<N>g` (paired with `--gpus=<count>`,
  a bare `--gres=gpumem:` alone doesn't reserve a GPU) gives a reliable VRAM floor as a second guard.
- `euler.ethz.ch` round-robins across multiple physical login nodes per SSH connection — a `tmux`
  session on one connection is invisible from the next. Pin to one login node hostname
  (`ssh <specific-node>`, e.g. `eu-login-14`) for anything that needs a persistent `tmux`/`salloc`
  session across multiple commands.
- Real workload memory for the quote/arb loss (`n_heldout=315`, `group_size=20`) needs the full
  96GB RTX Pro 6000 — an 80GB A100 OOMs (measured 78.2/79.25GB used before crash). Supervised loss
  (query = fixed 375-point native grid, no arb-grid blowup) is much cheaper and fits comfortably
  even at `group_size=128` (~41GB).

## Open items

- ~~Fix the butterfly tau-axis phase-anchoring bug (finding 9)~~ — resolved as a side effect of the
  session-2 grid rewrite (finding 13): `sample_arb_grid`'s `r_b` axis now always goes through
  `_jittered_axis`, which randomizes the start offset every call, not just the step.
- ~~Apply the `group_size=4`, long-step-count recipe to arb~~ — done, finding 14.
- **Retrain arb (`curvature_jitter`) under the finding-17 NLL fix** and re-check whether finding
  18's negative-IV outlier is actually resolved (`ssvi_opds_arb_curvature_jitter_gs4_15k_v2` in
  progress at time of writing) — this is the most important open item, since findings 14/18's
  numbers were all measured under the pre-fix loss.
- Once retrained under the fix, run the full-budget (72k-step) arb run that was cancelled
  pre-fix (finding 14) — apply the same "more steps helps" lever arb now shares with supervised.
- Evaluate `ssvi_supervised_gs4_24h` (144k steps, in progress) — does the "more steps helps" trend
  from finding 2 continue past 72k, or start flattening?
- Evaluate the queued `rho=0.5` run (finding 19) to fill in the middle of the rho curve.
- Re-test violators-only reduction *after* the grid-density fix, not before — finding 10 suggests
  it needs the density fix to pay off rather than backfiring; now that finding 9 is resolved this
  is unblocked.
- Run the off-grid-exposure-only ablation (finding 11) to properly isolate the arb loss's real
  contribution from the off-grid-query effect.
- Decide whether to switch `eval_arbitrage_fine` to median-based `mean_depth`/`worst_cell`
  reporting (finding 18) — deprioritized in favor of fixing the root cause first, revisit if the
  outlier persists after retraining.
- Real-data calibration (SPXW quotes) not revisited this session — last state predates the grid
  reparametrization; needs re-validation before reuse.
- Evaluate `ssvi_supervised_gs4_72k_heston_full` and the two `mix50` checkpoints on the same frozen
  Heston eval set (`datasets/eval/heston.pkl` + `heston_baselines.json` are cached on Euler, so each
  is a GPU pass only, refit columns free). Two of these runs were still training at time of writing.
- Complete the 2x2: SSVI refit vs Heston refit on *SSVI* data. Finding 27 only has the
  Heston-data row; the mirror row is what turns "SSVI is misspecified on Heston data" into a
  symmetric statement about family mismatch rather than a claim about SSVI specifically.
- Wire `fit_heston` into `run_finetuning.py`'s `_baselines` so mixture-trained runs get both refit
  columns by default, instead of only via the standalone `scripts/run_heston_eval.py`.
- Consider adding Bates and SABR families — Bates is nearly free given the QuantLib Heston setup
  (`BatesProcess`/`BatesModel`/`BatesEngine`), and a third family would let the misspecification
  floor in finding 27 be shown as a general effect rather than a single SSVI-vs-Heston pair.
