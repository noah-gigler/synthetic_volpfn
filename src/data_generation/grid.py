import numpy as np


def z_to_k(z, tau): return z * np.sqrt(tau)
def k_to_z(k, tau): return k / np.sqrt(tau)


class Grid:
    def __init__(self, cfg):
        rho = np.linspace(np.sqrt(cfg["ttm"]["min"]), np.sqrt(cfg["ttm"]["max"]), cfg["ttm"]["n_points"])
        self.ttms = rho**2
        self.zs = np.linspace(cfg["z"]["min"], cfg["z"]["max"], cfg["z"]["n_points"])
        self.rho = rho
        self.shape = (len(self.ttms), len(self.zs))

        TT, ZZ = np.meshgrid(self.ttms, self.zs, indexing="ij")   # ttm-major
        self.tau = TT.ravel()
        self.z = ZZ.ravel()
        self.k = z_to_k(self.z, self.tau)

    def features(self):
        return np.column_stack([self.z, self.tau])
