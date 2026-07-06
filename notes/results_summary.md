# Results summary (as of 2026-07-06)

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

## Known issue

**Stale-kernel eval artifact**: long-lived Jupyter kernels produced 10-50x-worse MAE at specific
(model, context-size) slots (e.g. noisyFT@60quotes = 0.037-0.050 in-kernel vs 0.0034 fresh),
consistent within a session, moving between sessions, never reproducible in a fresh process.
A within-process ordering experiment (same slot before/after a sweep-like history) showed *no*
effect, so a simple shape-keyed cache is ruled out; root cause unknown. Mitigation: restart the
kernel before evals, or run sweep slots in subprocesses. Do not trust in-kernel sweep numbers that
contradict training-val values.

## Open items

- Correlated quote noise (v2): current noise is independent per quote - the easiest kind to
  average away; real errors correlate across strikes/maturities (non-synchronous quotes, MM
  inventory). Likely to *strengthen* the model vs refit at large n.
- n=60 gap: WLS still wins the densest slot (~0.0018 vs ~0.0034 at m=1).
- Mid-input ablation: retrain the noisy model on mids to separate schema value from noise training.
- CRPS-only run for the 80%-interval calibration.
- Real-data transfer.
