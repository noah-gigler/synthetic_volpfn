# X = [k, tau, side], side in {-1: bid, +1: ask, 0: true}

import numpy as np
from scipy.stats import norm

from src.data_generation.data_preperation import (
    generate_surfaces,
    sample_context_sizes,
    sample_sparse_points,
)
from src.data_generation.grid import sample_arb_grid

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


def _corr_normal(rho, shape):
    # single-factor gaussian copula: rho=0 -> iid per quote, rho=1 -> one shared draw per surface
    z_surface = np.random.normal(0, 1)
    z_quote = np.random.normal(0, 1, shape)
    return np.sqrt(rho) * z_surface + np.sqrt(1 - rho) * z_quote


def half_spread(k, tau, sigma, noise_cfg, regime=1.0, rho=0.0):
    price, vega = _bs_otm_price_vega(k, tau, sigma)
    # proportional/inventory-risk component scales with market stress, tick effect does not
    s_price = regime * noise_cfg["beta"] * price + noise_cfg["tick"]
    s = s_price / np.maximum(vega, 1e-300)
    s = s * np.exp(noise_cfg["jitter_sigma"] * _corr_normal(rho, np.shape(s)))
    return np.minimum(s, noise_cfg["cap"])


def add_quote_noise(k, tau, sigma_true, noise_cfg, regime=1.0, rho=0.0):
    if regime == 0:
        return sigma_true, sigma_true

    # true IV at a uniform random position inside the spread
    s = half_spread(k, tau, sigma_true, noise_cfg, regime, rho)
    u = norm.cdf(_corr_normal(rho, np.shape(sigma_true)))
    bid = np.maximum(sigma_true - u * 2 * s, 1e-4)
    ask = bid + 2 * s
    return bid, ask


def _noisy_split(g, surfaces, k_idx, t_idx, regimes, noise_cfg, rho=0.0):
    # model feature is (z, tau, side); physical k = z·√τ is only used for BS spread pricing
    train, test = [], []
    for i in range(len(surfaces)):
        sigma = surfaces[i].ravel()
        idx = t_idx[i] * g.shape[1] + k_idx[i]
        zq, kq, tauq = g.z[idx], g.k[idx], g.tau[idx]

        bid, ask = add_quote_noise(kq, tauq, sigma[idx], noise_cfg, regimes[i], rho)

        X_train = np.column_stack([
            np.tile(zq, 2), np.tile(tauq, 2), np.repeat([BID, ASK], len(zq)),
        ])
        y_train = np.concatenate([bid, ask])

        X_test = np.column_stack([g.z, g.tau, np.full(len(sigma), TRUE)])

        train.append((X_train, y_train))
        test.append((X_test, sigma))

    return train, test


def _sample_regimes(noise_cfg, n, regime):
    if regime is None:
        return np.random.lognormal(0, noise_cfg["regime_sigma"], n)
    return np.full(n, float(regime))


def noisy_data_preparation(cfg, n, n_context, size_dist="uniform", regime=None, size_group=1, rho=0.0):
    # n_context counts quote locations (context holds 2*n_context rows)
    noise_cfg = cfg["noise"]
    g, surfaces = generate_surfaces(cfg, n)
    sizes = sample_context_sizes(n_context, n, dist=size_dist, group=size_group)
    k_idx, t_idx = sample_sparse_points(g.zs, g.ttms, sizes, n_samples=n)
    regimes = _sample_regimes(noise_cfg, n, regime)
    return _noisy_split(g, surfaces, k_idx, t_idx, regimes, noise_cfg, rho)


def quote_data_preparation(cfg, n, n_context, n_heldout, size_dist="uniform", regime=None, size_group=1, rho=0.0,
                            arb_grid_jitter=False):
    # decoupled query: [random arb grid (y=NaN, fresh every size_group chunk)] ++
    # [n_heldout quote rows (y=[bid,ask])]. True prices appear nowhere.
    noise_cfg = cfg["noise"]
    g, surfaces = generate_surfaces(cfg, n)
    sizes = sample_context_sizes(n_context, n, dist=size_dist, group=size_group)
    k_idx, t_idx = sample_sparse_points(g.zs, g.ttms, sizes + n_heldout, n_samples=n)
    regimes = _sample_regimes(noise_cfg, n, regime)

    train, test = [], []
    for start in range(0, n, size_group):
        arb_rows = sample_arb_grid(cfg, jitter=arb_grid_jitter)
        n_arb = len(arb_rows)
        for i in range(start, min(start + size_group, n)):
            sigma = surfaces[i].ravel()
            idx = t_idx[i] * g.shape[1] + k_idx[i]
            zq, kq, tauq = g.z[idx], g.k[idx], g.tau[idx]
            bid, ask = add_quote_noise(kq, tauq, sigma[idx], noise_cfg, regimes[i], rho)

            nc = len(idx) - n_heldout  # choice order is random -> first nc is a random split
            X_train = np.column_stack([
                np.tile(zq[:nc], 2), np.tile(tauq[:nc], 2), np.repeat([BID, ASK], nc),
            ])
            y_train = np.concatenate([bid[:nc], ask[:nc]])

            held_rows = np.column_stack([zq[nc:], tauq[nc:], np.full(n_heldout, TRUE)])
            X_test = np.vstack([arb_rows, held_rows])
            y_test = np.full((len(X_test), 2), np.nan)
            y_test[n_arb:] = np.column_stack([bid[nc:], ask[nc:]])

            train.append((X_train, y_train))
            test.append((X_test, y_test))

    return train, test


def make_quote_eval_set(cfg, n_surfaces, n_context, n_heldout, regime=None, size_group=1, rho=0.0):
    # drawn once -> frozen val set; size_group must match the training group_size so every
    # surface in a stacked group shares the same (randomized) arb-grid row count
    return quote_data_preparation(
        cfg, n_surfaces, n_context, n_heldout, regime=regime, size_group=size_group, rho=rho
    )


def make_noisy_stratified_eval_set(cfg, n_surfaces, context_sizes, regime=None, rho=0.0):
    # same n_surfaces at every size in context_sizes, size-major; noise drawn once (frozen)
    noise_cfg = cfg["noise"]
    g, surfaces = generate_surfaces(cfg, n_surfaces)
    regimes = _sample_regimes(noise_cfg, n_surfaces, regime)

    train, test = [], []
    for size in context_sizes:
        k_idx, t_idx = sample_sparse_points(g.zs, g.ttms, np.full(n_surfaces, size), n_samples=n_surfaces)
        tr, te = _noisy_split(g, surfaces, k_idx, t_idx, regimes, noise_cfg, rho)
        train += tr
        test += te

    return train, test
