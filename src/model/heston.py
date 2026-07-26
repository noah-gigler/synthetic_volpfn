"""Fit Heston parameters to a sparse set of quoted points via regularized least squares.

Parameter order matches the generator: (v0, kappa, theta, sigma, rho).
"""
import numpy as np
import QuantLib as ql
from scipy.optimize import least_squares

from src.data_generation.heston import _heston_engine, iv_at, sample_params

PARAM_NAMES = ("v0", "kappa", "theta", "sigma", "rho")

# unconstrained fit coordinates: log for the positive params, logit for -rho (the prior forces
# rho<0, see SSVI.sample_params). Removes the box entirely and puts all five on a comparable
# scale, so trf's trust region and its finite-difference steps mean the same thing for each -
# unscaled, kappa~2 and v0~0.04 differ by ~50x and the numerical Jacobian degrades.
def _to_u(p):
    v0, kappa, theta, sigma, rho = p
    return np.array([np.log(v0), np.log(kappa), np.log(theta), np.log(sigma),
                     np.log(-rho / (1 + rho))])


def _from_u(u):
    return np.array([np.exp(u[0]), np.exp(u[1]), np.exp(u[2]), np.exp(u[3]),
                     -1 / (1 + np.exp(-u[4]))])


def _prior_moments(cfg):
    # (mean, sd) of each fit coordinate under config.yaml's prior. Lognormal params are Gaussian
    # in log by construction; logit(-rho) is Gaussian by logitnormal's definition. v0 = v_bar*r
    # is a product of two independent lognormals, so its log-variance is the sum. sigma is
    # uniform (heston_prior), which no Gaussian penalty describes - left unpenalized.
    c = cfg
    m_vb, s_vb = np.log(c["v_bar"]["median"]), c["v_bar"]["sigma"]
    m_r, s_r = np.log(c["r"]["median"]), c["r"]["sigma"]
    rho_med = c["rho"]["median"]
    return np.array([
        [m_vb + m_r, np.hypot(s_vb, s_r)],
        [np.log(c["kappa"]["median"]), c["kappa"]["sigma"]],
        [m_vb, s_vb],
        [np.nan, np.inf],
        [np.log(rho_med / (1 - rho_med)), c["rho"]["sigma"]],
    ])


def heston_iv_pointwise(tau, k, params):
    v0, kappa, theta, sigma, rho = params
    tau, k = np.broadcast_arrays(np.asarray(tau, float), np.asarray(k, float))
    settlement = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = settlement
    try:
        engine = _heston_engine(float(v0), float(kappa), float(theta), float(sigma), float(rho),
                                settlement)
    except Exception:  # trf can propose a parameter vector QuantLib refuses to build at all
        return np.full(tau.shape, np.nan)

    out = np.empty(tau.size)
    for i, (t, kk) in enumerate(zip(tau.ravel(), k.ravel())):
        # unlike the generator's own calls, params here come from the optimizer, not the prior,
        # so the integral can diverge outright ("stdDev (nan) must be non-negative") - that's a
        # verdict on the proposal, not an error
        try:
            out[i] = iv_at(engine, settlement, float(kk), float(t))
        except Exception:
            out[i] = np.nan
    return out.reshape(tau.shape)


def _model_resid(k, ttm, y_w, params):
    iv = heston_iv_pointwise(ttm, k, params)
    r = iv**2 * ttm - y_w
    # context points are drawn from the valid mask, so every y_w here is a real quote: a model
    # that cannot price one has been proposed into a bad region. Scoring that as zero residual
    # (nan_to_num's default) would make an entirely unpriceable proposal look like a perfect
    # fit; charging it the full observed variance instead pushes trf back out.
    return np.where(np.isfinite(r), r, y_w)


def _residuals(u, k, ttm, y_w, weights, prior, lambda_prior, free_v0, v0_fixed):
    p = _from_u(u)
    if not free_v0:
        p[0] = v0_fixed
    r = _model_resid(k, ttm, y_w, p)
    if weights is not None:
        r = weights * r
    if lambda_prior <= 0:
        return r
    # Tikhonov rows toward the generating prior - the same regularization the inverse-problem
    # literature prescribes for this objective's flat valley, and it makes the fit a MAP
    # estimate under exactly the prior the surfaces were drawn from
    dev = (u - prior[:, 0]) / prior[:, 1]
    dev = np.nan_to_num(dev[np.isfinite(prior[:, 1])], nan=0.0)
    return np.concatenate([r, np.sqrt(lambda_prior) * dev])


def _init_params(cfg, n_restarts):
    med = np.array([cfg["v_bar"]["median"] * cfg["r"]["median"], cfg["kappa"]["median"],
                    cfg["v_bar"]["median"], np.sqrt(np.prod(list(cfg["heston_prior"]["sigma"].values()))),
                    -cfg["rho"]["median"]])
    inits = [med]
    for _ in range(max(0, n_restarts - 1)):
        inits.append(np.array([p[0] for p in sample_params(cfg, 1)]))
    return inits


# lambda_prior is only in MAP units when `weights` divides each residual by that quote's noise
# (the 1/(2*y*tau*s) weights the noisy pipeline already builds) - then both terms are
# dimensionless and lambda_prior=1 means "prior counts as much as the likelihood". Against
# unweighted residuals, which sit in raw total-variance units (~1e-3), the standardized prior
# rows (~1) outweigh the data by ~10^6 and the fit never leaves the prior median. Hence the
# default is off: callers with weights opt in explicitly.
def fit_heston(X, y, cfg, n_restarts=3, weights=None, lambda_prior=0.0, anchor_v0=True):
    # X columns are physical (k, tau), matching fit_ssvi
    k, ttm = X[:, 0], X[:, 1]
    y_w = y**2 * ttm  # fit in total-variance space

    # v0 <-> kappa is the worst of Heston's two documented degeneracies; anchoring v0 at the
    # shortest quoted maturity's ATM total variance (which the data pins down directly) breaks
    # it and drops the fit to four free parameters
    v0_fixed = None
    if anchor_v0:
        near = ttm <= np.quantile(ttm, 0.34)
        j = np.argmin(np.abs(k[near]))
        v0_fixed = float(np.clip(y[near][j] ** 2, 1e-4, 4.0))

    prior = _prior_moments(cfg)
    best, best_cost = None, np.inf
    for p0 in _init_params(cfg, n_restarts):
        if anchor_v0:
            p0 = np.asarray(p0, float).copy()
            p0[0] = v0_fixed
        res = least_squares(_residuals, _to_u(p0), method="trf",
                            args=(k, ttm, y_w, weights, prior, lambda_prior, not anchor_v0, v0_fixed))
        if res.cost < best_cost:
            best, best_cost = res, res.cost

    p = _from_u(best.x)
    if anchor_v0:
        p[0] = v0_fixed
    return tuple(p), best_cost


def heston_data_cost(X, y, params, weights=None):
    # data term only (no prior rows), so the objective is comparable between a fitted parameter
    # vector and the true generating one - the diagnostic that separates "noise moved the
    # minimum" (ill-posedness) from "the optimizer failed" (a bug)
    k, ttm = X[:, 0], X[:, 1]
    r = _model_resid(k, ttm, y**2 * ttm, params)
    if weights is not None:
        r = weights * r
    return 0.5 * float(np.sum(r**2))


def predict_heston(params, ttms, ks):
    return heston_iv_pointwise(ttms, ks, params)
