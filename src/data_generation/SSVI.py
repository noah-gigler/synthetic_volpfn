import numpy as np
import yaml

# generate log normal squeezed into 0 - 1
def logitnormal(median, sigma, size):
    mu = np.log(median / (1 - median))
    x = np.random.normal(mu, sigma, size)
    return 1 / (1 + np.exp(-x))

# Max η satisfying Theorem 4.2 (Gatheral & Jacquier 2013) butterfly arbitrage conditions:
#   1. θφ(θ)(1+|ρ|) < 4  →  η(1+|ρ|) < 4
#   2. θφ(θ)²(1+|ρ|) ≤ 4 →  η ≤ 2/sqrt(B(θ_min)·(1+|ρ|))
def max_eta(rho, v_bar, v0, kappa, gamma, t_min):
    bound1 = 4 / (1 + np.abs(rho))
    theta_min = v_bar * t_min + (v0 - v_bar) / kappa * (1 - np.exp(-kappa * t_min))
    B_min = theta_min ** (1 - 2 * gamma) / (1 + theta_min) ** (2 - 2 * gamma)
    bound2 = 2 / np.sqrt(B_min * (1 + np.abs(rho)))

    return np.minimum(bound1, bound2)


def sample_params(cfg, n):
    rho = -logitnormal(cfg["rho"]["median"], cfg["rho"]["sigma"], n)
    gamma = logitnormal(cfg["gamma"]["median"], cfg["gamma"]["sigma"], n)

    v_bar = np.random.lognormal(np.log(cfg["v_bar"]["median"]), cfg["v_bar"]["sigma"], n)
    ratio = np.random.lognormal(np.log(cfg["r"]["median"]), cfg["r"]["sigma"], n)
    kappa = np.random.lognormal(np.log(cfg["kappa"]["median"]), cfg["kappa"]["sigma"], n)
    v0 = v_bar * ratio


    eta = np.random.uniform(0, max_eta(rho, v_bar, v0, kappa, gamma, cfg["ttm"]["min"]), n)

    return rho, eta, gamma, v_bar, v0, kappa

def ssvi(ttms, ks, rho, eta, gamma, v_bar, v0, kappa):
    # dimensional casting for vectorization; returns (n, dim(ttms), dim(ks last axis))
    # ks: (n_k,) shared across ttm, or (n_ttm, n_k) wedge (k = z·√τ)
    ttms = ttms[None, :, None]
    ks = ks[None, None, :] if ks.ndim == 1 else ks[None, :, :]
    rho, eta, gamma, v_bar, v0, kappa = [x[:, None, None] for x in (rho, eta, gamma, v_bar, v0, kappa)]

    thetas = v_bar * ttms + (v0 - v_bar)/kappa * (1 - np.exp(-kappa * ttms))        # heston like term structure
    phis = eta / (thetas**gamma * (1 + thetas)**(1 - gamma))                        # modified power law
    w = thetas/2 * (1 + rho*phis*ks + np.sqrt((phis*ks + rho)**2 + (1 - rho**2)))   # SSVI parametrization

    return np.sqrt(w/ttms)


if __name__ == "__main__":
    ttms = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    ks   = np.linspace(-0.5, 0.5, 10)
    cfg = yaml.safe_load(open("config.yaml"))
    rho, eta, gamma, v_bar, v0, kappa = sample_params(cfg, n=1000)

    surfaces = ssvi(ttms, ks, rho, eta, gamma, v_bar, v0, kappa) 

    print(surfaces)


