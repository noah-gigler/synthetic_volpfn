# Results summary (as of 2026-07-14)

Chronology of finetuning experiments and what each established. Checkpoints in `checkpoints/<run>/`;
all evals on synthetic SSVI surfaces from the `config.yaml` prior, MAE vs the clean truth on the
full 15x25 grid unless stated otherwise.

## Runs

| run | context | training | key outcome |
|---|---|---|---|
| `ssvi_fixed_context_20` | fixed 20 quotes | 2k steps, batch 1 | -77% RMSE vs baseline at n=20, but weak off its size |
| `ssvi_dynamic_context_3_30` | log-uniform 3-30 | 2k steps, batch 1 | big small-n gains, regressed at n>=15 (over-weighted sparse regime; very spiky training) |
| `ssvi_uniform_context_3_30` (formerly `..._accum4_long`) | uniform 3-30 | 15k steps, batch 4 | dominates everywhere clean: beats fixed-20 at n=20 (MAE 0.0010 vs 0.0018-ish) while best at small n |
| `ssvi_noisy_uniform_3_60` | bid/ask schema, uniform 3-60 quotes | 15k steps, batch 4 | the noisy-quote model; see below |
| `ssvi_quote_n20` / `ssvi_quote_n20_long` | bid/ask, fixed 20 quotes, **quote loss (no true prices)** | 500 / 15k steps, batch 4 | headline: truth-free parity with supervision; see below |
| `ssvi_quote_uniform_3_60` | bid/ask, uniform 3-60 quotes, quote loss | 15k steps, batch 4 | truth-free parity at *all* sizes with one model; edge cliff above n~55; see below |

## Established findings

1. **Gradient accumulation (batch_size=4) fixed training noise.** Batch-1 epoch losses swung
   0.12-47 (single degenerate tiny-context surfaces yanking the weights); with accumulation the
   same setup stays within ~1 order of magnitude and val curves are smooth.
2. **Epochs are bookkeeping.** Fresh surfaces every epoch => total optimizer steps is the real
   budget; the 15k-step runs converge (val flat by ~epoch 285/300 with cosine anneal).
3. **Prefer `final.pt`.** Mean-val best-checkpoint selection is dominated by the smallest-context
   losses (they are 10-50x larger); large-context performance keeps improving until the anneal
   ends. `best.pt` at the mean-val optimum was measurably worse at n=40.
4. **SSVI refit identifiability (clean quotes)**: exact recovery at n>=6 points (= parameter
   count; needs >=3 maturities and off-ATM strikes). At n=5 the refit reaches ~0.006 RMSE, which
   upper-bounds the Bayes floor => the clean model's n=5 error (~0.019) is mostly a trainable gap,
   not irreducible ambiguity. On clean data the refit is an oracle for n>=6 - TabPFN's case there
   is only uncertainty quantification.
5. **Bid-ask noise model** (`src/data_generation/noise.py`): vega-based half-spread
   (tick + beta*price)/vega, per-surface lognormal regime m, per-quote lognormal jitter, 5-vol-pt
   cap; truth uniform *inside* the quoted spread (asymmetric, no mid observed). ~0.4 vol pts ATM at
   m=1, matching empirical SPX levels. Schema: X=[k,tau,side], side -1=bid/+1=ask/0=true(query).
6. **Headline noisy result: the finetuned model beats weighted SSVI refit (mids + observable
   1/spread weights) in the sparse regime, and the crossover moves up with noise.**
   MAE, corrected sweep (25 test surfaces/slot):
   - m=0.5: FT wins n<=8, WLS wins beyond.
   - m=1: FT wins n<=10 (e.g. n=5: 0.017 vs 0.029), ties at 40 (0.0021 both), WLS wins 15-20 and 60.
   - m=2: FT wins every size through 40 (n=20: 0.0040 vs 0.0059); WLS only wins at 60.
   Mid-input TabPFN (baseline and clean-finetuned) is far behind both under noise. Exactly the
   Bayesian prediction: weaker likelihood => learned prior worth more.
7. **Calibration** (central-interval coverage, noisy model): ~nominal at 20% and 50% levels,
   slightly thin at 80% (0.68-0.86). Candidate fixes if needed: CRPS-only loss (drop MSE term) or
   post-hoc conformal widening.
8. **Inside-spread fraction is ~100% and must be** - the generator places truth inside the spread
   by construction, so this metric only flags pathologies; it cannot distinguish denoising from
   mid-interpolation. Use the visual slice plots for that.

9. **HEADLINE — quote loss: supervised-quality surfaces without ever observing a true price.**
   Training loss (`src/model/quote_loss.py`): interval-censored NLL `-log P(bid <= y <= ask)`
   (bar-distribution CDF difference) at 15 *held-out* quote locations per surface + calendar/
   butterfly hinges on the raw-space pointwise mean over the full grid (the substitute for the
   arb-free-truth signal supervised training gets implicitly). Provider
   (`quote_data_preparation` in `noise.py`): context = bid/ask rows at 20 quotes, query = full
   grid, y_query = [bid, ask] at held-out locations / NaN elsewhere - truth appears nowhere,
   including checkpoint selection (val = same proxy loss on frozen surfaces).
   After 15k steps (`ssvi_quote_n20_long`, lambda_cal = lambda_bf = 1.0), MAE vs hidden truth
   at N=20 quotes, 25 test surfaces:

   | m | quote FT | supervised (`ssvi_noisy_uniform_3_60`) | refit WLS | baseline (mids) | mid noise floor |
   |---|---|---|---|---|---|
   | 0.5 | 0.0027 | 0.0025 | 0.0020 | 0.0120 | 0.0061 |
   | 1.0 | 0.0031 | 0.0031 | 0.0030 | 0.0146 | 0.0079 |
   | 2.0 | 0.0048 | 0.0051 | 0.0063 | 0.0160 | 0.0119 |

   Reading: parity with the fully supervised model at every noise level (nominally ahead at
   m=2), ties WLS at m=1 and beats it at m=2, ~2.5x below the raw mid noise floor, and **0%
   calendar/butterfly violations at all regimes** (the short 500-step run still had 12%/4% at
   m=0.5 - fixed by training budget alone, no lambda retuning). The supervised comparison is
   fair-to-conservative: the multi-size supervised model beat the fixed-20 specialist at n=20
   in the clean experiments, so the ceiling is the strongest available.
   Implication: the truth label is nearly redundant given quotes + no-arb + the learned prior,
   at this context size and noise model. Since the quote loss needs only observable data, the
   same training loop runs on real market quotes unchanged (data-provider swap) - the sim-to-real
   plan is now "pretrain on synthetic, calibrate on real quotes", with no simulator-fidelity cap.
   Notes: interval NLL has a nonzero floor (truth's position inside the spread is irreducibly
   uncertain), so train/val plateau at a positive value at convergence; per-surface loss scale
   varies with the noise regime (tight intervals = harder), which makes train_loss look noisier
   than the optimization actually is.

10. **Variable-context quote loss: the fixed-20 parity generalizes — one truth-free model covers
    3-60 quotes at supervised quality.** `ssvi_quote_uniform_3_60` (same loss/lambdas as the n20
    run, n_context uniform 3-60, n_heldout=15, 15k steps; val flat by ~epoch 260). Sweep at
    m in {0.5, 1, 2}, n in {3,...,60}, 50 test surfaces/slot: quote FT within 0.0002-0.0008 MAE
    of the supervised model at *every* slot (~5-20% relative, nominally behind everywhere —
    largest at n=60). vs WLS refit reproduces finding 6's noise-dependent crossover; new
    datapoint: **WLS blows up at n=5 under noise** (mean MAE 0.067 at m=1, 0.037 at m=2 — heavy
    tail of unidentifiable 5-point fits) while FT stays ~0.017 — the learned prior fails
    gracefully where least-squares fails catastrophically. Denoising confirmed: FT ~2-2.5x below
    the mid noise floor at n>=20.
11. **Sparse-context arb violations are generic, not a quote-loss deficiency.** n=5, m=1,
    100 surfaces, fresh process, identical draws: quote FT 5% cal / 8% butterfly vs supervised
    4% / 6%, and largely the *same* surfaces violate (7 IDs in both lists). Magnitudes are real
    (g down to -0.3/-0.65, calendar dw/dt down to -5e-3), and violators have ~2x the MAE of the
    average surface — arb appears exactly on the draws the model gets wrong anyway. So the hinge
    already matches implicit arb-free-truth supervision; to go below that, crank lambda (only the
    quote model has the knob) or post-hoc SSVI projection at very sparse n.
12. **Edge-of-training-range cliff at dense contexts — train to 60, deploy to <=50.** Inside-spread
    fraction (m=1, 25 surfaces/slot, fresh process): quote FT 99.2% @ n=50 -> 80% @ 55 -> 51% @ 58;
    supervised 91% -> 80% -> 79%; vanilla baseline flat 65-69% at all sizes (no cliff — uniformly
    mediocre denoising). The cliff exists only in models finetuned with the uniform(3,60) context
    distribution => training-distribution edge effect, not spread-NLL-specific (though the quote
    model collapses harder at 58, and both FT models' truth-MAE is non-monotonic 40 -> 60 in the
    sweep). Broad-based — all 25 surfaces < 90% at n=58 — so not filterable; keep ~10 quotes of
    margin below the training max, or widen the training range if dense contexts are needed.
    Flip side: away from the edge the quote model respects spreads *better* than the supervised
    one (99.2% vs 91.3% at n=50, worst surface 96% vs 58%) — direct interval training pays.

## Real-data experiments (SPXW EOD 2023, 2026-07-13/14)

First runs on real market quotes. Data: `datasets/processed/spxw.parquet` (248 EOD SPXW surfaces,
2023, one 16:14 ET snapshot/day, ~5-6k OTM quotes each; built by `src/real_data/preprocess_surfaces.py`
from the Databento pipeline). Dataloader `src/real_data/dataloader.py`: temporal train/val/test split
(163/22/63 days), ATM-weighted context sampling, `build_task`/`make_real_eval_set` emit the
`quote_data_preparation` contract so the quote/arb loss + `finetune` run unchanged. No true IV exists
on real data, so MAE is vs the **market mid** (a noisy proxy), and checkpoint selection stays truth-free.

**Metric note:** `check_arbitrage` returns `.any()` per surface, so reported "arb %" = fraction of
surfaces with **>=1** violating cell, *not* fraction of cells.

### Runs

| run | config | outcome |
|---|---|---|
| `real_v1` | mean reduction, lambda=1 (arb ~off), 50ep, coarse grid | MAE 0.90% @N40, inside 29.8% — best fit but **~50% fine-grid arb: unusable surface** |
| `real_v2` | violators, lambda=100, 50ep | over-regularized: MAE 2.10%, std inflated |
| `real_v3` | violators, lambda=10, 150ep, **coarse** grid | MAE 1.79% looks great but **~50% arb on the honest fine grid** (coarse-trained) |
| `real_v4_curv{5,20,50}` | fine jittered grid + curvature, 50ep | curvature marginal (see finding 16) |
| `real_v5_bf10_c0` | fine jittered grid, violators, lambda_bf=10, 150ep | **winner**: MAE 1.88% @N40, ~5% fine-grid arb, beats SSVI at N>=10 |
| `real_v5_bf10_c5` / `bf3_c0` | +curvature / lambda_bf=3 | c5 over-smooths (inside 12%); bf3 arb rebounds to ~9% |

Headline comparison @N=40, honest fine grid (MAE/inside at held quotes, grid-independent):

| model | MAE% | inside% | fine-grid arb bf% |
|---|---|---|---|
| SSVI oracle | 2.38 | 8.1 | 0 (by constr.) |
| **v5 bf10_c0** | **1.88** | 24.7 | **~5** |
| v5 bf3_c0 | 1.71 | 26.1 | ~9 |
| v5 bf10_c5 | 2.15 | 12.2 | ~3.6 |
| v1 (arb off) | 0.90 | 29.8 | **~50** |

### Established findings

13. **The coarse 15x25 arb grid is falsely optimistic — violations don't converge.** Evaluating a
    fixed surface at 1x/2x/3x/4x resolution: incidence 6% -> 62% -> 100% -> 100%, violating-cell
    count grows ~proportionally to grid size (constant *fraction* of cells non-convex at every scale).
    TabPFN predicts each (z,tau) quasi-independently, so the surface is wiggly at all scales and a
    coarse grid steps over most of it. Real SPXW has ~50-100 strikes/expiry -> the honest grid is
    ~2-3x; at that density the coarse "clean" models are ~50% violating. **Measure arb on the fine
    grid.**
14. **Violators reduction >> mean.** Penalizing mean violation *depth over violating cells only*
    (`v.sum()/(v>0).sum()`) beats averaging over the whole lattice: at equal lambda it cuts arb ~3x
    for near-zero NLL cost; to match it, `mean` needs ~100-1000x larger lambda and pays more NLL.
    `mean` dilutes the gradient and vanishes as violations get rare (teaches-to-the-grid). Now the
    **only** reduction in `quote_arb_loss` (the param was removed).
15. **Fine + jittered grid is the real fix.** Training the penalty on a 2x grid whose axis nodes are
    re-jittered each epoch (stochastic collocation — prevents teaching-to-the-grid; `Grid(..., jitter=)`)
    drops honest fine-grid arb from **~50% (v3, coarse-trained) to ~5% (v5)** while holding MAE. Jitter
    can only move whole axis slices (the finite-difference stencil needs a rectangular lattice), so it
    perturbs the `zs`/`rho` axis vectors, not individual nodes.
16. **Curvature penalty removed — marginal and costly.** A `mean(w_zz^2)` roughness prior traded ~1%
    arb for ~0.3% MAE (and craters inside-spread 25%->12%). Not worth it; deleted from the code.
17. **lambda_bf=10 is right.** bf=3 too weak (arb rebounds to ~9%); bf=100 over-regularizes (MAE
    doubles, std inflates). bf=10 gives ~5% arb at MAE 1.88%.
18. **Full anneal matters for the fine-grid objective.** Unlike the coarse/synthetic runs (val flat by
    ~epoch 30), the harder fine-grid arb objective was still improving at epoch 50 — 150 epochs
    improved both MAE and arb. Data is the ceiling (163-day pool; val saturates ~epoch 30 only for the
    easy objective).
19. **HEADLINE — arb is nearly free in *true* accuracy; the real-data MAE "regression" is
    noise-overfitting, not lost accuracy.** Synthetic truth validation (`synth_val.py`, 40ep, true IV
    known), true-MAE %:

    | N | quote_arb (bf=10) | quote_noarb (bf=0) | supervised |
    |---|---|---|---|
    | 5 | 3.76 | 3.52 | 2.93 |
    | 20 | 1.30 | 1.17 | 0.89 |
    | 40 | 1.00 | 0.94 | 0.64 |

    arb: quote_arb **0% at all N**; quote_noarb and even **supervised** violate (6-12%). So (a)
    enforcing arb costs only ~0.06-0.24% *true*-MAE while eliminating violations — the v1 0.9%->1.9%
    real gap is the model correctly *shedding noisy-mid overfitting*, not losing real accuracy; (b)
    **only the arb penalty gives arb-free surfaces — supervision alone does not**; (c) the real,
    unavoidable gap is truth-free vs supervised (~0.3% @N40, grows with context) = the cost of interval
    censoring, and it's an *unreachable* ceiling (no true-IV label exists in the market; training on
    mids = v1 = overfitting). Possible partial recovery: exploit bid/ask **size**/microstructure to
    sharpen the interval likelihood (open item).

**Working model: `real_v5_bf10_c0`** — arb-free (~5% fine-grid) at true-MAE near the truth-free ceiling,
beats the SSVI oracle at N>=10.

## Open items

- Real-data calibration: swap `quote_data_preparation` for a real-quote provider (chain ->
  context/held-out split); pretrain synthetic, finetune with the quote loss on market data.
- Correlated quote noise (v2): current noise is independent per quote - the easiest kind to
  average away; real errors correlate across strikes/maturities (non-synchronous quotes, MM
  inventory). Likely to *strengthen* the model vs refit at large n.
- n=60 gap: explained by the edge cliff (finding 12) - WLS wins the densest slot partly because
  the FT models degrade there. If dense contexts matter, retrain with a wider range (e.g. 3-90)
  and re-sweep; prediction: the FT-vs-WLS crossover at n=60 moves in FT's favor.
- Mid-input ablation: retrain the noisy model on mids to separate schema value from noise training.
- CRPS-only run for the 80%-interval calibration (supervised model runs slightly thin tails).
- Quote-loss watch item: interval NLL alone leaves within-spread location underdetermined at
  quote points; overlapping quotes + arb + prior pinned it here, revisit if real data behaves
  differently (fallback: margin/quantile-outside penalty).
- (real data) More trading days: 163-day 2023 pool is the ceiling (val saturates ~epoch 30). Pull
  2020-22/2024 EOD SPXW to grow it — the only lever that moves MAE *and* arb together (Databento cost).
- (real data) Microstructure interval likelihood: bid/ask *size* / traded side may carry where the
  true IV sits in the spread — could recover part of the ~0.3% truth-free-vs-supervised gap (finding 19).
- (real data) Sparse regime (N=5) is worst on both MAE and arb; targeted hard-day/small-context
  emphasis before spending on more data.
