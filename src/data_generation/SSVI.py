import numpy as np
import yaml

# generate log normal squeezed into 0 - 1
def logitnormal(median, sigma, size):
    mu = np.log(median / (1 - median))
    x = np.random.normal(mu, sigma, size)
    return 1 / (1 + np.exp(-x))

def sample_params(cfg, n):
    rho = -logitnormal(cfg["rho"]["median"], cfg["rho"]["sigma"], n)
    gamma = logitnormal(cfg["gamma"]["median"], cfg["gamma"]["sigma"], n)
    max_nu = 2/(1 + np.abs(rho))
    nu = np.random.uniform(0, max_nu, n) # enforces arbitrage free condition

    v_bar = np.random.lognormal(np.log(cfg["v_bar"]["median"]), cfg["v_bar"]["sigma"], n)
    ratio = np.random.lognormal(np.log(cfg["r"]["median"]), cfg["r"]["sigma"], n)
    kappa = np.random.lognormal(np.log(cfg["kappa"]["median"]), cfg["kappa"]["sigma"], n)
    v0 = v_bar * ratio

    return rho, nu, gamma, v_bar, v0, kappa

def ssvi(ttms, log_moneyness, rho, nu, gamma, v_bar, v0, kappa):
    # dimensional casting for vectorization
    # returns (n, ttms, ks)
    ttms = ttms[None, :, None]
    log_moneyness = log_moneyness[None, None, :]
    rho, nu, gamma, v_bar, v0, kappa = [x[:, None, None] for x in (rho, nu, gamma, v_bar, v0, kappa)]

    thetas = v_bar * ttms + (v0 - v_bar)/kappa * (1 - np.exp(-kappa * ttms))
    phis   = nu / (thetas**gamma * (1 + thetas)**(1 - gamma))
    w = thetas/2 * (1 + rho*phis*log_moneyness + np.sqrt((phis*log_moneyness + rho)**2 + (1 - rho**2)))

    return np.sqrt(w/ttms)


if __name__ == "__main__":
    ttms = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    ks   = np.linspace(-0.5, 0.5, 10)
    cfg = yaml.safe_load(open("config.yaml"))["ssvi_prior"]
    rho, nu, gamma, v_bar, v0, kappa = sample_params(cfg, n=1000)

    surfaces = ssvi(ttms, ks, rho, nu, gamma, v_bar, v0, kappa) 

    print(surfaces)


