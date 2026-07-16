# Results summary (as of 2026-07-16)

Everything before the `z`-reparametrization + widened grid (`config.yaml`: `z∈[-1.5,0.5]`,
`ttm_min=0.02`, commit `1c4c12e7` onward) is archived in git history, not repeated here — those
numbers are on a materially different (narrower, `k`-parametrized) task and aren't a fair
comparison point anymore. This doc covers the current grid/config only.

## Current best model

**`ssvi_supervised_gs4_12h`** — supervised (true-IV) loss, `group_size=4, batch_size=16`,
2400 epochs × 30 steps/epoch = **72,000 optimizer steps** (~12h on an RTX Pro 6000 Blackwell).
Checkpoint + logs in `checkpoints/ssvi_supervised_gs4_12h/`.

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
11. **Direct measurement confirms this is a coverage problem, not a lambda-magnitude problem.**
    Evaluating `g_fn` on the *exact* training collocation points (not an eval-side grid) gave
    literally 0/2200 violating points for a real trained checkpoint — the model satisfies the
    constraint perfectly wherever it's actually checked during training, while a broader fine grid
    still shows ~1-2% violation. Raising lambda further cannot help this: there's no gradient to
    amplify when the sampled points are already all satisfied. The fix has to be *coverage*
    (denser/differently-structured tau sampling), not loss weight.
12. **Comparison caveat: supervised (non-arb) models are never trained on off-grid query points at
    all** — their query is always the fixed native 25x15 grid, every epoch. Arb-loss models get
    (sparse, phase-biased) off-grid exposure via `sample_arb_grid`. So the earlier read that "arb
    loss only modestly beats plain supervised on fine-grid violation rate" (supervised: 50% of
    surfaces / 1.5-2.1% cells vs arb: 0.9-1.2% cells) is **not a clean comparison** — the arb
    models are being tested on exactly the kind of location they were partially trained for, the
    supervised ones never saw such points at all. An ablation (supervised loss + off-grid query
    exposure, no arb penalty) would be needed to isolate the two effects. Not yet run.

## Bugs found and fixed this session

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

- Fix the butterfly tau-axis phase-anchoring bug (finding 9) — highest-priority remaining lever
  for arb, per finding 11's direct evidence that it's a coverage problem, not a lambda problem.
- Re-test violators-only reduction *after* the grid-density fix, not before — finding 10 suggests
  it needs the density fix to pay off rather than backfiring.
- Run the off-grid-exposure-only ablation (finding 12) to properly isolate the arb loss's real
  contribution from the off-grid-query effect.
- Apply the `group_size=4`, long-step-count recipe (the supervised winner) to the quote/arb-loss
  model too — all arb runs so far have used comparatively few steps (~500 epochs at
  `group_size=20/batch_size=40` ≈ 6-8k steps), not the 72k-step budget that closed the gap for
  supervised.
- Real-data calibration (SPXW quotes) not revisited this session — last state predates the grid
  reparametrization; needs re-validation before reuse.
