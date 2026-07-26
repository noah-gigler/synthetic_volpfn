import sys
import types

import numpy as np
import yaml
import QuantLib as ql

if "_testcapi" not in sys.modules:  # vollib imports DBL_MIN/MAX. hacky fix (see src/real_data/quotes.py)
    _shim = types.ModuleType("_testcapi")
    _shim.DBL_MIN, _shim.DBL_MAX = sys.float_info.min, sys.float_info.max
    sys.modules["_testcapi"] = _shim
from vollib.black.implied_volatility import implied_volatility as black_iv

from src.data_generation.SSVI import sample_params as ssvi_sample_params

# QuantLib's AnalyticHestonEngine (semi-analytic, Fourier/Gauss-Kronrod integration under the
# hood) prices each surface's grid via a plain per-point loop, one engine per surface. A
# hand-rolled vectorized quadrature was tried first and benchmarked SLOWER (~2x) at this
# project's actual batch sizes despite being "vectorized": it needed 256 quadrature nodes to
# stay accurate everywhere (a 96-node version had branch-cut/under-resolution bugs - see git
# history), and large per-batch intermediate arrays hit real memory-bandwidth limits, an effect
# that gets worse with batch size while QuantLib's per-point loop doesn't. QuantLib is also
# already battle-tested (used for the initial cross-validation of the abandoned quadrature
# approach) rather than a from-scratch implementation with its own subtle sign/branch bugs to
# maintain. r=q=0 throughout, matching this project's forward-relative k=ln(K/F) convention.
_QL_INTEGRATION_ORDER = 64


def _heston_engine(v0, kappa, theta, sigma, rho, settlement):
    dc = ql.Actual365Fixed()
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement, 0.0, dc))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement, 0.0, dc))
    s0 = ql.QuoteHandle(ql.SimpleQuote(1.0))
    process = ql.HestonProcess(r_ts, q_ts, s0, v0, kappa, theta, sigma, rho)
    return ql.AnalyticHestonEngine(ql.HestonModel(process), _QL_INTEGRATION_ORDER)


def _call_price_one(engine, settlement, k, tau):
    # undiscounted call price / F (r=q=0, F=S=1), one (k, tau) point
    days = max(1, int(round(tau * 365)))
    exercise_date = settlement + ql.Period(days, ql.Days)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(np.exp(k)))
    exercise = ql.EuropeanExercise(exercise_date)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    return option.NPV()


def iv_at(engine, settlement, k, tau):
    days = max(1, int(round(tau * 365)))
    tau_actual = days / 365.0
    call_price = _call_price_one(engine, settlement, k, tau)
    is_call_otm = k > 0
    target = call_price if is_call_otm else call_price - 1.0 + np.exp(k)
    cp = "c" if is_call_otm else "p"
    if target > 0:
        try:
            return black_iv(target, 1.0, np.exp(k), 0.0, tau_actual, cp)
        except Exception:  # negative-noise below intrinsic / above max / no convergence
            pass
    return np.nan


def sample_params(cfg, n, max_tries=50):
    # v_bar/r/kappa/rho in config.yaml's ssvi_prior are already historically-calibrated-to-SPX
    # Heston-style parameters, not SSVI-specific: SSVI.py's own `thetas` term-structure formula
    # (v_bar*ttm + (v0-v_bar)/kappa*(1-exp(-kappa*ttm))) IS the Heston mean-reverting-variance
    # formula, and rho is already forced negative there (real equity index skew, "leverage
    # effect"). Reusing them here - rather than a second, disconnected uniform-range prior -
    # keeps both surface families anchored to the same real-market beliefs, and avoids the
    # symmetric-rho mistake a generic prior would make (half of the surfaces would then get the
    # wrong-signed skew for the target real-data domain). Only sigma (vol-of-vol) is genuinely
    # new - SSVI has no vol-of-vol parameter at all, being a static (non-stochastic-vol) model.
    rho, _eta, _gamma, theta, v0, kappa = ssvi_sample_params(cfg, n)
    p = cfg["heston_prior"]["sigma"]
    sigma = np.random.uniform(p["min"], p["max"], n)

    # Feller (2*kappa*theta >= sigma^2) keeps variance strictly positive; not required for
    # arbitrage-freeness (Heston is arb-free regardless) but keeps the pricing integral in a
    # numerically well-behaved region. Resample sigma alone for violators (same rejection-style
    # pattern as SSVI.py's max_eta), since sigma is the only parameter here without an existing,
    # externally-motivated prior to preserve.
    for _ in range(max_tries):
        violates = 2 * kappa * theta < sigma**2
        if not violates.any():
            break
        sigma[violates] = np.random.uniform(p["min"], p["max"], violates.sum())

    # for (kappa, theta) draws where 2*kappa*theta < sigma_min**2, no sigma in the configured
    # range can satisfy Feller at all (resampling loop above can never converge for these -
    # confirmed empirically at ~0.16% of draws under this config's ranges) - deterministic
    # fallback clip guarantees Feller always holds rather than silently leaving violators
    still_violates = 2 * kappa * theta < sigma**2
    sigma[still_violates] = 0.999 * np.sqrt(np.maximum(2 * kappa[still_violates] * theta[still_violates], 1e-8))

    return v0, kappa, theta, sigma, rho


def heston(ttms, ks, v0, kappa, theta, sigma, rho):
    # dimensional casting matches SSVI.ssvi(): returns (n, dim(ttms), dim(ks last axis))
    ttms_b = ttms[None, :, None]
    ks_b = ks[None, None, :] if ks.ndim == 1 else ks[None, :, :]
    tau, k = np.broadcast_arrays(ttms_b, ks_b)  # (1 or n, n_ttm, n_k) -> broadcast to full shape below
    n = len(v0)
    tau = np.broadcast_to(tau, (n,) + tau.shape[1:])
    k = np.broadcast_to(k, (n,) + k.shape[1:])

    settlement = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = settlement

    iv_flat = np.full(k.size, np.nan)
    pos = 0
    for i in range(n):
        engine = _heston_engine(float(v0[i]), float(kappa[i]), float(theta[i]),
                                 float(sigma[i]), float(rho[i]), settlement)
        for tau_ij, k_ij in zip(tau[i].ravel(), k[i].ravel()):
            iv_flat[pos] = iv_at(engine, settlement, k_ij, tau_ij)
            pos += 1

    return iv_flat.reshape(k.shape)


if __name__ == "__main__":
    ttms = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    ks = np.linspace(-0.5, 0.5, 10)
    cfg = yaml.safe_load(open("config.yaml"))
    v0, kappa, theta, sigma, rho = sample_params(cfg, n=1000)

    surfaces = heston(ttms, ks, v0, kappa, theta, sigma, rho)
    print(surfaces.shape, "nan frac:", np.isnan(surfaces).mean())
