# X = [k, tau, side], side in {-1: bid, +1: ask, 0: true}

import numpy as np
from scipy.stats import norm

from src.data_generation.data_preperation import (
    generate_surfaces,
    sample_context_sizes,
    sample_sparse_points,
)

BID, ASK, TRUE = -1.0, 1.0, 0.0


def _bs_otm_price_vega(k, tau, sigma):
    # OTM option price and vega per unit forward (F=1, r=0)
    sqrt_tau = np.sqrt(tau)
    d1 = -k / (sigma * sqrt_tau) + sigma * sqrt_tau / 2
    d2 = d1 - sigma * sqrt_tau
    vega = norm.pdf(d1) * sqrt_tau
    call = norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
    put = call + np.exp(k) - 1  # put-call parity with F = 1
    return np.where(k >= 0, call, put), vega


def half_spread(k, tau, sigma, noise_cfg, regime=1.0):
    price, vega = _bs_otm_price_vega(k, tau, sigma)
    s_price = regime * (noise_cfg["tick"] + noise_cfg["beta"] * price)
    s = s_price / np.maximum(vega, 1e-300)
    s = s * np.exp(np.random.normal(0, noise_cfg["jitter_sigma"], np.shape(s)))
    return np.minimum(s, noise_cfg["cap"])


def add_quote_noise(k, tau, sigma_true, noise_cfg, regime=1.0):
    # true IV at a uniform random position inside the spread; regime=0 -> bid=ask=true
    s = half_spread(k, tau, sigma_true, noise_cfg, regime)
    u = np.random.uniform(0, 1, np.shape(sigma_true))
    bid = np.maximum(sigma_true - u * 2 * s, 1e-4)
    ask = bid + 2 * s
    return bid, ask


def _noisy_split(ks, ttms, surfaces, k_idx, t_idx, regimes, noise_cfg):
    TT, KK = np.meshgrid(ttms, ks, indexing="ij")
    k_flat, tau_flat = KK.ravel(), TT.ravel()

    train, test = [], []
    for i in range(len(surfaces)):
        sigma = surfaces[i].ravel()
        idx = t_idx[i] * len(ks) + k_idx[i]
        kq, tauq = k_flat[idx], tau_flat[idx]

        bid, ask = add_quote_noise(kq, tauq, sigma[idx], noise_cfg, regimes[i])

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


def noisy_data_preparation(cfg, n, n_context, size_dist="uniform", regime=None, size_group=1):
    # n_context counts quote locations (context holds 2*n_context rows)
    noise_cfg = cfg["noise"]
    ttms, ks, surfaces = generate_surfaces(cfg, n)
    sizes = sample_context_sizes(n_context, n, dist=size_dist, group=size_group)
    k_idx, t_idx = sample_sparse_points(ks, ttms, sizes, n_samples=n)
    regimes = _sample_regimes(noise_cfg, n, regime)
    return _noisy_split(ks, ttms, surfaces, k_idx, t_idx, regimes, noise_cfg)


def quote_data_preparation(cfg, n, n_context, n_heldout, size_dist="uniform", regime=None, size_group=1):
    # query y = [bid, ask] at n_heldout held-out quote locations, NaN elsewhere; no true prices
    noise_cfg = cfg["noise"]
    ttms, ks, surfaces = generate_surfaces(cfg, n)
    sizes = sample_context_sizes(n_context, n, dist=size_dist, group=size_group)
    k_idx, t_idx = sample_sparse_points(ks, ttms, sizes + n_heldout, n_samples=n)
    regimes = _sample_regimes(noise_cfg, n, regime)

    TT, KK = np.meshgrid(ttms, ks, indexing="ij")
    k_flat, tau_flat = KK.ravel(), TT.ravel()

    train, test = [], []
    for i in range(n):
        sigma = surfaces[i].ravel()
        idx = t_idx[i] * len(ks) + k_idx[i]
        kq, tauq = k_flat[idx], tau_flat[idx]
        bid, ask = add_quote_noise(kq, tauq, sigma[idx], noise_cfg, regimes[i])

        nc = len(idx) - n_heldout  # choice order is random -> first nc is a random split
        X_train = np.column_stack([
            np.tile(kq[:nc], 2), np.tile(tauq[:nc], 2), np.repeat([BID, ASK], nc),
        ])
        y_train = np.concatenate([bid[:nc], ask[:nc]])

        X_test = np.column_stack([k_flat, tau_flat, np.full(len(sigma), TRUE)])
        y_test = np.full((len(sigma), 2), np.nan)
        y_test[idx[nc:]] = np.column_stack([bid[nc:], ask[nc:]])

        train.append((X_train, y_train))
        test.append((X_test, y_test))

    return train, test


def make_quote_eval_set(cfg, n_surfaces, n_context, n_heldout, regime=None):
    # drawn once -> frozen val set
    return quote_data_preparation(cfg, n_surfaces, n_context, n_heldout, regime=regime)


def make_noisy_stratified_eval_set(cfg, n_surfaces, context_sizes, regime=None):
    # same n_surfaces at every size in context_sizes, size-major; noise drawn once (frozen)
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
