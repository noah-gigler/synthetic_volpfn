# Bid-ask quote noise for synthetic surfaces, plus the noisy (bid/ask) data providers.
#
# Layered spread model, deliberately not recoverable from (k, tau) alone:
#   vega-based structural half-spread  x  per-surface regime  x  per-quote jitter,
# with the true IV at a uniform random position inside the quoted spread, so the
# spread is asymmetric around the truth and no mid is ever observed.
#
# Noisy input schema: X = [k, tau, side] with side -1 = bid, +1 = ask, 0 = true.
# Each sampled quote location contributes two context rows (bid and ask); query rows
# are the full grid with side = 0 and clean targets. side = 0 sits between the two
# labeled values so plain interpolation already biases toward "inside the spread".

import numpy as np
from scipy.stats import norm

from src.data_generation.data_preperation import (
    generate_surfaces,
    sample_context_sizes,
    sample_sparse_points,
)

BID, ASK, TRUE = -1.0, 1.0, 0.0


def _bs_otm_price_vega(k, tau, sigma):
    """Black-Scholes OTM option price and vega per unit forward (F=1, r=0)."""
    sqrt_tau = np.sqrt(tau)
    d1 = -k / (sigma * sqrt_tau) + sigma * sqrt_tau / 2
    d2 = d1 - sigma * sqrt_tau
    vega = norm.pdf(d1) * sqrt_tau
    call = norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
    put = call + np.exp(k) - 1  # put-call parity with F = 1
    return np.where(k >= 0, call, put), vega


def half_spread(k, tau, sigma, noise_cfg, regime=1.0):
    """IV half-spread: price half-spread (tick + beta*price, the standard fixed +
    proportional cost decomposition) scaled by the regime, converted to IV units via
    vega, per-quote lognormal jitter, capped (liquidity filter proxy - nobody quotes
    8-sigma wings tighter than the cap). The tick term is what widens the wings."""
    price, vega = _bs_otm_price_vega(k, tau, sigma)
    s_price = regime * (noise_cfg["tick"] + noise_cfg["beta"] * price)
    s = s_price / np.maximum(vega, 1e-300)
    s = s * np.exp(np.random.normal(0, noise_cfg["jitter_sigma"], np.shape(s)))
    return np.minimum(s, noise_cfg["cap"])


def add_quote_noise(k, tau, sigma_true, noise_cfg, regime=1.0):
    """True IV sits at a uniform random position inside the quoted spread.
    Returns (bid, ask); regime=0 collapses to bid = ask = true."""
    s = half_spread(k, tau, sigma_true, noise_cfg, regime)
    u = np.random.uniform(0, 1, np.shape(sigma_true))
    bid = np.maximum(sigma_true - u * 2 * s, 1e-4)
    ask = bid + 2 * s
    return bid, ask


def _noisy_split(ks, ttms, surfaces, k_idx, t_idx, regimes, noise_cfg):
    """Build (train, test) with bid/ask context rows and clean full-grid query."""
    TT, KK = np.meshgrid(ttms, ks, indexing="ij")
    k_flat, tau_flat = KK.ravel(), TT.ravel()

    train, test = [], []
    for i in range(len(surfaces)):
        sigma = surfaces[i].ravel()
        idx = t_idx[i] * len(ks) + k_idx[i]
        kq, tauq = k_flat[idx], tau_flat[idx]

        bid, ask = add_quote_noise(kq, tauq, sigma[idx], noise_cfg, regimes[i])

        # two rows per quote: first all bids, then all asks (same (k, tau) order)
        X_train = np.column_stack([
            np.tile(kq, 2), np.tile(tauq, 2), np.repeat([BID, ASK], len(kq)),
        ])
        y_train = np.concatenate([bid, ask])

        X_test = np.column_stack([k_flat, tau_flat, np.full(len(sigma), TRUE)])

        train.append((X_train, y_train))
        test.append((X_test, sigma))

    return train, test


def _sample_regimes(noise_cfg, n, regime):
    if regime is None:
        return np.random.lognormal(0, noise_cfg["regime_sigma"], n)
    return np.full(n, float(regime))


def noisy_data_preparation(cfg, n, n_context, size_dist="uniform", regime=None):
    """Bid/ask counterpart of data_preparation; same (train, test) contract, so it plugs
    into finetune() via functools.partial. `n_context` counts quote locations (a context
    holds 2*n_context rows). `regime=None` samples a lognormal noise regime per surface;
    a scalar fixes it for all surfaces (0 -> noiseless quotes)."""
    noise_cfg = cfg["noise"]
    ttms, ks, surfaces = generate_surfaces(cfg, n)
    sizes = sample_context_sizes(n_context, n, dist=size_dist)
    k_idx, t_idx = sample_sparse_points(ks, ttms, sizes, n_samples=n)
    regimes = _sample_regimes(noise_cfg, n, regime)
    return _noisy_split(ks, ttms, surfaces, k_idx, t_idx, regimes, noise_cfg)


def make_noisy_stratified_eval_set(cfg, n_surfaces, context_sizes, regime=None):
    """Mirror of make_stratified_eval_set for the bid/ask schema: the *same* n_surfaces
    surfaces (and per-surface noise regimes) at every size in `context_sizes`, size-major.
    Noise is drawn once at construction, so the eval task is frozen."""
    noise_cfg = cfg["noise"]
    ttms, ks, surfaces = generate_surfaces(cfg, n_surfaces)
    regimes = _sample_regimes(noise_cfg, n_surfaces, regime)

    train, test = [], []
    for size in context_sizes:
        k_idx, t_idx = sample_sparse_points(ks, ttms, np.full(n_surfaces, size), n_samples=n_surfaces)
        tr, te = _noisy_split(ks, ttms, surfaces, k_idx, t_idx, regimes, noise_cfg)
        train += tr
        test += te

    return train, test
