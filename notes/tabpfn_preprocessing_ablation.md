# TabPFN preprocessing ablation

Pretrained (non-finetuned) TabPFN V3, `n_estimators=1`, 50 held-out SSVI surfaces, `n_context=20`.

## Default pipeline (per row, for `(k, tau) -> IV`)

1. `RemoveConstantFeaturesStep` — no-op
2. `SquashingScaler` (robust median-center + IQR-scale + soft-clip to `[-3,3]`) on k, tau — replaces raw values
3. `AddSVDFeaturesStep` (SVD, quarter-components) — appends 1 extra column: rank-1 SVD of the squashed (k, tau)
4. `AddFingerprintFeaturesStep` — appends 1 extra column: hash of the row, meant to disambiguate duplicate rows
5. `ShuffleFeaturesStep` — random column-order permutation (irrelevant at 2-3 features)
6. y standardized to z-score; no power-transform (`safepower` branch not selected at `n_estimators=1`)

## Results

| Squashing | SVD | Fingerprint | RMSE | vs default |
|---|---|---|---|---|
| ✓ | ✓ | off | **0.0094** | **-18%** |
| ✓ | ✓ | on (stock default) | 0.0114 | — |
| ✓ | off | on | 0.0161 | +41% |
| off | ✓ | off | 0.0115 | +1% |
| ✓ | off | off | 0.0196 | +72% (worst) |

Also tested, both worse/neutral:
- Forcing `safepower` y-transform: 0.0121 (+6%)
- Using the alternate default branch (`quantile_uni`) instead of squashing+SVD: 0.0140 (+23%)
- Ensembling both branches (`n_estimators=2`): 0.0108, still worse than squashing+SVD alone

## Conclusions

- **Squashing and SVD are each independently load-bearing** — removing either one alone is harmful, removing both together is even worse. Keep both, unchanged from default.
- **Fingerprint feature is genuinely useless here and safe to remove** — it's a random hash-based column meant to disambiguate duplicate rows in real tabular data; irrelevant for a dense, non-duplicated (k, tau) grid, and empirically it hurts (adds noise the attention mechanism has to process).
- `safepower` y-transform and the `quantile_uni` branch: not helpful for this task, but not tested as safe to actively force off (leave at default since they're inactive anyway at `n_estimators=1`).
