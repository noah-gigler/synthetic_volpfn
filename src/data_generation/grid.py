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


def _jittered_axis(lo, hi, step, n_points=None):
    # like np.arange(lo, hi, step) but with a random start-of-domain offset (not just step)
    # so the far edge of the domain isn't structurally excluded every single call; when
    # n_points is given, the offset is chosen so the count stays fixed regardless of jitter
    span = hi - lo
    n_intervals = (n_points - 1) if n_points is not None else int(span / step)
    slack = span - n_intervals * step
    offset = np.random.uniform(0, max(slack, 0.0))
    return lo + offset + np.arange(n_intervals + 1) * step


def arb_grid_shape(cfg):
    # the 3 axis counts that are always fixed (cfg-derived, never randomized in count) - lets
    # quote_arb_loss recover row boundaries by pure position/arithmetic, no marker column needed:
    # rows are always [n_rb*n_zb butterfly] + [(n_rc-1)*n_zc cal_lo] + [(n_rc-1)*n_zc cal_hi],
    # and n_rb = (query_len - n_heldout - 2*(n_rc-1)*n_zc) / n_zb
    z_lim = (cfg["z"]["min"], cfg["z"]["max"])
    rho_lim = (np.sqrt(cfg["ttm"]["min"]), np.sqrt(cfg["ttm"]["max"]))
    n_zb = len(np.arange(z_lim[0], z_lim[1], 0.01))
    n_rc = len(np.arange(rho_lim[0], rho_lim[1], 0.02))
    n_zc = len(np.arange(z_lim[0], z_lim[1], 0.1))
    return n_zb, n_rc, n_zc


def sample_arb_grid(cfg, jitter=False, r_b_step_range=(0.075, 0.125)):
    # random rectilinear grids for the arb penalty, à la operator-deep-smoothing's
    # Loss._build_grids: butterfly = coarse random rho x dense fixed z (needs d/dz, d2/dz2);
    # calendar = fine fixed rho x coarse random z, as ~n_rc-1 adjacent-rho row pairs (needs
    # only a direct two-point total-variance comparison, no derivative). All rows carry
    # side=TRUE (row type is recovered by position/count in quote_arb_loss, see
    # arb_grid_shape - never smuggled through a feature column). Returns X_rows [N, 3] with
    # columns [z, tau, side].
    # r_b always gets a randomized start offset (not just step) - otherwise the far domain
    # edge is structurally never checked; jitter=True additionally offsets z_b/r_c, which are
    # otherwise identical every single call (step stays fixed either way, only offset moves,
    # so n_zb/n_rc/n_zc - all fixed via arb_grid_shape - stay unchanged by jitter).
    # r_b_step_range controls how many rho rows r_b gets (still a fresh random step per call,
    # just narrower/coarser depending on the caller); training uses the wide cheap default,
    # eval passes a narrow range centered on the step that gives real-density-matched coverage.
    rho_lim = (np.sqrt(cfg["ttm"]["min"]), np.sqrt(cfg["ttm"]["max"]))
    z_lim = (cfg["z"]["min"], cfg["z"]["max"])
    n_zb_fixed, n_rc_fixed, n_zc_fixed = arb_grid_shape(cfg)

    r_b = _jittered_axis(*rho_lim, np.random.uniform(*r_b_step_range))
    z_b = _jittered_axis(*z_lim, 0.01, n_points=n_zb_fixed) if jitter else np.arange(z_lim[0], z_lim[1], 0.01)
    TT, ZZ = np.meshgrid(r_b**2, z_b, indexing="ij")   # tau-major
    but_rows = np.column_stack([ZZ.ravel(), TT.ravel(), np.zeros(TT.size)])

    r_c = _jittered_axis(*rho_lim, 0.02, n_points=n_rc_fixed) if jitter else np.arange(rho_lim[0], rho_lim[1], 0.02)
    n_rc = len(r_c)
    # z_c's count must always be n_zc_fixed (not just under jitter=True) - quote_arb_loss
    # recovers row boundaries purely by arithmetic on this fixed count, unconditionally
    z_c = _jittered_axis(*z_lim, np.random.uniform(0.075, 0.125), n_points=n_zc_fixed)
    r_lo, r_hi = r_c[:-1], r_c[1:]
    z_hi = z_c[None, :] * (r_lo / r_hi)[:, None]           # rescaled so both slices share k
    cal_lo_rows = np.column_stack([
        np.tile(z_c, n_rc - 1), np.repeat(r_lo**2, len(z_c)), np.zeros((n_rc - 1) * len(z_c)),
    ])
    cal_hi_rows = np.column_stack([
        z_hi.ravel(), np.repeat(r_hi**2, len(z_c)), np.zeros((n_rc - 1) * len(z_c)),
    ])

    return np.vstack([but_rows, cal_lo_rows, cal_hi_rows])
