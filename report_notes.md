# Report notes

Working notes for the semester thesis writeup (8 ECTS, deadline ~1 week out). Reconciles
`VolSmoothing_with_TabPFN_proposal.pdf` against what's actually in the repo. Proposal is not fully
binding — several extensions became the core of the project, several base-plan items were dropped.
Report should include an explicit short "deviations from proposal" paragraph rather than silently
diverge.

## The story (three acts + a case study)

1. Clean synthetic SSVI data, supervised loss — does finetuning beat/match the SSVI refit oracle?
   Yes, at moderate/large context and non-trivial noise; batching/step-count engineering
   (`group_size`) was the single biggest lever, bigger than loss design.
2. Noisy bid/ask quotes, supervised loss — holds up under realistic quote noise, with a clean,
   mechanistically-explained crossover (SSVI wins back at large context + low noise, its
   asymptotic-efficiency regime) and a real finding on noise-correlation (`rho`) mismatch.
3. Truth-free quote/arb loss — learns without ever seeing a true price, via interval-censored NLL +
   calendar/butterfly penalties. Headline result: at N=20 quotes, matches supervised MAE with ~0%
   arb violations. This is the strongest, most novel contribution — the proposal itself flagged
   arb-penalty-in-the-loss as uncertain/hard (ICL not autodiff-differentiable in k); this project
   found a working path via grid-based, position-recovered penalties on the decoded raw-space mean.
4. Real-data (SPXW) case study — weakest, most open thread. Frame as diagnosed-but-unresolved, not
   a pillar. Current best inside-spread ~21% vs an old-pipeline reference of 57.8%; several
   hypotheses tested this week (y-scale source, robust/global squashing, warm-start instability,
   noise-correlation mismatch) with mixed/negative results, each with a clear mechanistic diagnosis
   even when the fix didn't fully work (e.g. warm-start collapse traced to bar-distribution
   bin-edge mismatch after switching target rescale constants, fixed via lower warm-start lr).

## Proposal vs. actual: section-by-section

**§2 Problem setup** — done as specified (in-context regression, randomized sparse context/query
split), with one deliberate, undocumented-in-proposal deviation: model is fed standardized
`z = k/√τ`, not raw `k`. Ablation justifies this (raw-k hurt sparse-context regime) — needs its own
paragraph, not a silent swap.

**§3 Approach** — partially done.
- ~~Missing: the SSVI+Heston mixture.~~ **Closed (2026-07-26).** Heston generation, a Heston refit
  baseline, and mixture sampling now exist; `ssvi_supervised_gs4_15k_heston_full` is trained purely
  on Heston and evaluated against both refits on a pure-Heston eval set (results_summary findings
  22–29). The §4 "prior generalizes across families" framing now has a real result behind it: the
  SSVI refit hits a ~0.5% misspecification floor on Heston data that no amount of context or noise
  reduction moves, while the finetuned model beats *both* refits at small context and ties the
  correctly-specified Heston refit at large context. Two `mix50` checkpoints still training.
- Loss changed: proposal's eq. (1) is plain IV-MSE. Actual loss is TabPFN's own bar-distribution
  (bucket cross-entropy / CRPS-style) loss throughout — better suited to the UQ goal (full
  predictive distribution vs. point + separate variance), but eq. (1) doesn't describe what was
  built. Needs its own subsection.

**§4 Evaluation** — mixed.
- Point estimate (IV-RMSE/MAE vs SSVI): done extensively, strongest pillar.
- vs. ODS/SANOS/Ackerer/HyperIV: not done, no other baseline implemented. State explicitly as
  out of scope rather than omit silently.
- Uncertainty metrics: partial. Have `quantile_coverage` (levels 0.2/0.5/0.8) and
  `inside_spread_fraction` (bid/ask coverage proxy, not in original proposal). Missing: CRPS as a
  *reported* metric (only exists as a training-loss ingredient, `nll`), pinball loss, mean interval
  width, PIT calibration histograms. Real gap given proposal calls UQ "the primary differentiator" —
  addressing this is the next work item (see below).
- Arbitrage violation rate: done, taken further than proposed (fine-grid, characterized failure
  modes, several real bugs found/fixed along the way — good process narrative).
- OOD generalization (train SSVI, eval Heston): not done, follows from missing mixture.
- Real market data, masked-quote protocol: not done as specified (lightweight zero-shot,
  no retraining). Project skipped straight to full real-data finetuning (extension) instead.
  Worth adding the cheap zero-shot masked-quote check too if time allows — nearly free (single
  eval-only pass, synthetic checkpoint, no training) and gives a "does the prior transfer at all
  before real-data training" data point currently missing from the story.

**§5 Extensions — where the actual project lives**
- Bid-ask noise: done, deeper than proposed (layered spread model: regime/jitter/tick structure,
  `rho` correlation ablation with a real explained mechanism).
- Real market training: done (`src/real_data/`, Databento OPRA SPXW instead of OptionMetrics),
  currently the weakest/least-finished thread.
- Arbitrage penalty in the loss: done — best result, more than proposal asked for.
- Richer mixture (SABR, VolGANs, neural SDE): not done, consistent with missing base mixture.

## Suggested report structure

1. Introduction — motivation, research question, contribution summary.
2. Background — SSVI, arbitrage-free conditions, TabPFN/ICL, why finetuning not zero-shot.
3. Method — SSVI generation + prior, z-parametrization + justification, sparse-context sampling,
   batched finetuning loop (`group_size` as an engineering contribution), bid-ask noise model,
   three loss variants (clean supervised / noisy supervised / truth-free quote-arb).
4. Results — one subsection per act above (clean synthetic, noisy synthetic, truth-free arb loss),
   plus a short "what didn't work" subsection (n_estimators=2 instability, k-feature ablation
   revert, violators-only loss backfiring) — good process narrative, not padding.
5. Real-data case study (SPXW) — explicitly framed as preliminary/exploratory, honest current
   numbers, diagnosed open gap.
6. Discussion — when/why a learned prior beats a correctly-specified parametric model (rho
   noise-collapse mechanism is the best concrete insight here); limitations (synthetic-to-real gap,
   compute constraints, single-family mixture, arb-loss NLL edge case history).
7. Conclusion + future work.

## What to cut/compress given the 1-week deadline

- Don't chase a real-data breakthrough this week — write up the honest, diagnosed-but-open numbers.
  A clearly-explained negative result beats a rushed unverified claim from an unfinished run.
- Session-1-style debugging trivia (SSVI refit z-vs-k bug, `_stackable` query-shape bug) → footnote
  or appendix, not main narrative.
- Full `group_size` throughput sweep table → one sentence + table in appendix, it's infrastructure
  not science.

## UQ metrics — done (2026-07-25)

Added `eval_uncertainty` to `src/evaluation/surface_eval.py`: CRPS, pinball loss (default levels
0.05/0.95), central-interval width (default 90%), and raw PIT values, all read off the bar
distribution's own `cdf`/`icdf` on the existing batched forward pass (no retraining, no new model
output). CRPS has no closed form for a piecewise-uniform bar distribution, so it's estimated via
the standard quantile-average identity `CRPS(F,y) = 2 * E_tau~U(0,1)[pinball_tau(y, F^-1(tau))]`
(Gneiting & Raftery 2007) on a 39-point quantile grid. PIT values are returned raw (not just a
summary stat) so the report can plot a PIT histogram — under good calibration they should be
~Uniform(0,1); skew toward 0/1 means over/under-confident intervals, skew toward the middle means
under-confident (too-wide) intervals.

Found and fixed a real bug in `BarDistribution.cdf` while wiring this up: its 1-D-`ys` branch is
meant for "shared evaluation points broadcast across the batch" (e.g. plotting a fixed CDF grid for
every row), not "one true value per row" — since our per-row `y` happens to also be 1-D with length
matching the query dim, it silently mis-broadcast into a `(Q,Q)` matrix product instead of a
positional match. Fixed by giving `y` an explicit trailing singleton dim so it takes `cdf`'s
assert-checked positional branch instead. Worth a footnote in the report's "bugs found" section —
this one wasn't in the training path, only in the new eval code, so it never affected any of the
existing MAE/arb numbers.

Wired into `scripts/run_finetuning.py`'s `run_eval` (synthetic eval only) — every eval run now also
writes a CRPS/pinball/interval-width table to `eval.txt`/`eval.json`, plus raw PIT arrays per
(regime, n_ctx) slot to `pit.npz` for histogram plotting. **Deliberately not added to
`run_real.py`**: real data has no true IV (`CLAUDE.md`: "true prices appear nowhere" in the
real-data quote/arb setup), so PIT/CRPS against a point truth would be meaningless there —
`inside_spread_fraction`/`ins%` (already existing) is the correct interval-based UQ proxy for real
data, since it checks against bid/ask bounds instead of a nonexistent point value.

Not yet done: nothing has been *re-evaluated* under this yet — the numbers in every existing
`eval.txt` on Euler predate this addition. Next `--eval-only` pass on any checkpoint (synthetic
supervised/arb) will produce real CRPS/pinball/PIT numbers for the report; the PIT histogram plot
itself (reading `pit.npz`) still needs to be written, likely as a small local plotting script once
at least one eval has been re-run with the new code.
