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


# arb-grid row-kind markers (see sample_arb_grid); distinct from noise.py's BID/ASK/TRUE
BUTTERFLY, CAL_LOW, CAL_HIGH = 2.0, 3.0, 4.0


def sample_arb_grid(cfg):
    # random rectilinear grids for the arb penalty, à la operator-deep-smoothing's
    # Loss._build_grids: butterfly = coarse random rho x dense fixed z (needs d/dz, d2/dz2);
    # calendar = fine fixed rho x coarse random z, as ~n_rc-1 adjacent-rho row pairs (needs
    # only a direct two-point total-variance comparison, no derivative). Returns
    # (X_rows [N, 3] with columns [z, tau, side], n_zb, n_rc).
    rho_lim = (np.sqrt(cfg["ttm"]["min"]), np.sqrt(cfg["ttm"]["max"]))
    z_lim = (cfg["z"]["min"], cfg["z"]["max"])

    r_b = np.arange(rho_lim[0], rho_lim[1], np.random.uniform(0.075, 0.125))
    z_b = np.arange(z_lim[0], z_lim[1], 0.01)
    n_zb = len(z_b)
    TT, ZZ = np.meshgrid(r_b**2, z_b, indexing="ij")   # tau-major
    but_rows = np.column_stack([ZZ.ravel(), TT.ravel(), np.full(TT.size, BUTTERFLY)])

    r_c = np.arange(rho_lim[0], rho_lim[1], 0.02)
    n_rc = len(r_c)
    z_c = np.arange(z_lim[0], z_lim[1], np.random.uniform(0.075, 0.125))
    r_lo, r_hi = r_c[:-1], r_c[1:]
    z_hi = z_c[None, :] * (r_lo / r_hi)[:, None]           # rescaled so both slices share k
    cal_lo_rows = np.column_stack([
        np.tile(z_c, n_rc - 1), np.repeat(r_lo**2, len(z_c)), np.full((n_rc - 1) * len(z_c), CAL_LOW),
    ])
    cal_hi_rows = np.column_stack([
        z_hi.ravel(), np.repeat(r_hi**2, len(z_c)), np.full((n_rc - 1) * len(z_c), CAL_HIGH),
    ])

    return np.vstack([but_rows, cal_lo_rows, cal_hi_rows]), n_zb, n_rc
