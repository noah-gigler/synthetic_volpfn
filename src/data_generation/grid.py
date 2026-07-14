import numpy as np


def z_to_k(z, tau): return z * np.sqrt(tau)
def k_to_z(k, tau): return k / np.sqrt(tau)


def _jitter(a, frac, rng):
    # shift interior nodes by up to frac of the smaller neighbouring gap (endpoints fixed,
    # frac<0.5 keeps the axis strictly monotone so the finite-difference stencil stays valid)
    a = a.copy()
    gaps = np.diff(a)
    a[1:-1] += rng.uniform(-frac, frac, len(a) - 2) * np.minimum(gaps[:-1], gaps[1:])
    return a


class Grid:
    def __init__(self, cfg, z_mult=1, ttm_mult=1, jitter=0.0, rng=None):
        n_ttm = (cfg["ttm"]["n_points"] - 1) * ttm_mult + 1
        n_z = (cfg["z"]["n_points"] - 1) * z_mult + 1
        rho = np.linspace(np.sqrt(cfg["ttm"]["min"]), np.sqrt(cfg["ttm"]["max"]), n_ttm)
        zs = np.linspace(cfg["z"]["min"], cfg["z"]["max"], n_z)
        if jitter and rng is not None:
            rho = _jitter(rho, jitter, rng)   # jitter in rho so the calendar stencil stays clean
            zs = _jitter(zs, jitter, rng)
        self.ttms = rho**2
        self.zs = zs
        self.rho = rho
        self.shape = (len(self.ttms), len(self.zs))

        TT, ZZ = np.meshgrid(self.ttms, self.zs, indexing="ij")   # ttm-major
        self.tau = TT.ravel()
        self.z = ZZ.ravel()
        self.k = z_to_k(self.z, self.tau)

    def features(self):
        return np.column_stack([self.z, self.tau])
