"""Fit SSVI parameters to a sparse set of quoted points via least squares.

Parameter order matches the generator: (rho, eta, gamma, v_bar, v0, kappa).
"""
import numpy as np
from scipy.optimize import least_squares

from src.data_generation.SSVI import ssvi, sample_params

PARAM_NAMES = ("rho", "eta", "gamma", "v_bar", "v0", "kappa")

_EPS = 1e-8
_BOUNDS = (
    np.array([-1.0 + _EPS, _EPS, _EPS, _EPS, _EPS, _EPS]),
    np.array([1.0 - _EPS, 50.0, 1.0 - _EPS, 5.0, 5.0, 50.0]),
)


def ssvi_w_pointwise(ttm, k, params):
    rho, eta, gamma, v_bar, v0, kappa = params
    theta = v_bar * ttm + (v0 - v_bar) / kappa * (1 - np.exp(-kappa * ttm))
    phi = eta / (theta**gamma * (1 + theta) ** (1 - gamma))
    return theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + (1 - rho**2)))


def ssvi_iv_pointwise(ttm, k, params):
    return np.sqrt(ssvi_w_pointwise(ttm, k, params) / ttm)


def _init_params(cfg, n_restarts):
    inits = [np.array([
        -cfg["rho"]["median"], 1.0, cfg["gamma"]["median"],
        cfg["v_bar"]["median"], cfg["v_bar"]["median"] * cfg["r"]["median"], cfg["kappa"]["median"],
    ])]
    for _ in range(max(0, n_restarts - 1)):
        inits.append(np.array([p[0] for p in sample_params(cfg, 1)]))
    return [np.clip(p, _BOUNDS[0], _BOUNDS[1]) for p in inits]


def fit_ssvi(X, y, cfg, n_restarts=3, weights=None):
    """X is (n, 2) columns [k, tau], y is implied vol. Returns (params, cost).

    `weights` (optional, per point) scale the total-variance residuals - for noisy
    quotes with IV noise sd s_i use weights 1/(2*y_i*tau_i*s_i) (delta method:
    dw = 2*sigma*tau*dsigma)."""
    k, ttm = X[:, 0], X[:, 1]
    y_w = y**2 * ttm  # fit in total-variance space

    def residuals(p):
        r = ssvi_w_pointwise(ttm, k, p) - y_w
        return r if weights is None else weights * r

    best = None
    for p0 in _init_params(cfg, n_restarts):
        res = least_squares(residuals, p0, bounds=_BOUNDS, method="trf")
        if best is None or res.cost < best.cost:
            best = res

    return tuple(best.x), best.cost


def predict_ssvi(params, ttms, ks):
    return ssvi(ttms, ks, *[np.array([p]) for p in params])[0]
