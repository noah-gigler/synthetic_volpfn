# Finetuning TabPFN for Implied Volatility Surface Smoothing

Master's semester project at ETH Zürich.

Fitting an implied volatility surface means reconstructing a whole surface from a handful of quoted option prices. The standard approach fits a parametric form like SSVI by least squares, which works well when quotes are plentiful but degrades when they are sparse, and gives no honest uncertainty estimate.

This project asks whether a **tabular foundation model** can do better. [TabPFN](https://github.com/PriorLabs/TabPFN) is pretrained to do regression in-context, so it reads a set of observed points and predicts unobserved ones in a single forward pass with no per-dataset fitting. Here it is finetuned on synthetically generated volatility surfaces, so the "prior" it learns is the shape of arbitrage-free surfaces rather than generic tabular data.

## Approach

Training data is generated, not collected. SSVI (and Heston) surfaces are sampled from a parameter prior, arbitrage bounds enforced at sampling time, and each surface is split into a sparse context of quotes and a query grid to predict. The model is finetuned across thousands of such surfaces, so it learns to interpolate and extrapolate the family rather than any single surface.

Three settings, in increasing order of realism:

1. **Clean supervised** — context is exact IVs, loss against true IVs. Establishes whether finetuning can match the SSVI refit oracle.
2. **Noisy quotes, supervised** — a layered bid/ask spread model (structural vega-based half-spread, per-surface regime, per-quote jitter) replaces clean points. The true IV sits at a random position inside the spread, so no mid is ever observed.
3. **Truth-free** — the interesting one. Training never sees a true price. The loss is an interval-censored likelihood on bid/ask bounds (how much probability mass the model puts inside the quoted spread) plus calendar and butterfly hinge penalties for arbitrage violations.

Setting 3 matters because it is the only one that could be trained on real market data, where a "true" implied vol does not exist.

## Results

- The truth-free loss reaches the same accuracy against held-out true IVs as the fully supervised model at 20 quotes per surface, with essentially no arbitrage violations, despite never training on a true price.
- Against the SSVI refit, the finetuned model wins in the sparse regime, where the least-squares fit is underdetermined. The refit wins back at large context and low noise, its asymptotic-efficiency regime.
- Trained on Heston surfaces, the SSVI refit hits a misspecification floor that more data does not fix, while the finetuned model beats both refits at small context and matches a correctly specified Heston refit at large context. The learned prior is not tied to the family it was trained on.
- Uncertainty is evaluated properly rather than assumed: CRPS, pinball loss, interval width, and PIT calibration, all read off the model's predictive distribution.

`notes/tabpfn_preprocessing_ablation.md` has a component ablation of TabPFN's preprocessing for this task, including the finding that its fingerprint feature actively hurts on a dense grid.

## Status

The synthetic side is complete. Training on **real SPXW option quotes** (via Databento OPRA) is in progress and is the open thread: the model transfers worse than the synthetic results suggest, and the gap is diagnosed but not closed.

## Layout

| Path | Contents |
|---|---|
| `src/data_generation/` | SSVI and Heston surface generation, grid, sparse-context sampling, bid/ask noise model |
| `src/model/` | Finetuning loop, quote/arbitrage loss, SSVI least-squares baseline, preprocessing |
| `src/evaluation/` | Arbitrage checks, error metrics, uncertainty metrics (CRPS, pinball, PIT) |
| `src/real_data/` | Databento SPXW ingestion |
| `scripts/` | Run entry points for finetuning and evaluation |
| `notebooks/` | Exploratory work per experiment |
| `config.yaml` | Parameter prior and grid definition, shared by generation and training |

## Running

```bash
uv sync
uv run python -m src.data_generation.data_preperation   # smoke test the generator
uv run python scripts/run_finetuning.py --help
```

Training runs on GPU; the synthetic data generation and SSVI baselines are CPU only. Real-data scripts hit a paid Databento subscription and are not runnable without credentials.
