import numpy as np

from src.data_generation.data_preperation import grid_from_cfg


def check_arbitrage(iv, ttms, ks, tol=-1e-10):
    w = iv**2 * ttms[:, None]

    dw_dt = np.diff(w, axis=-2)
    cal_violations = (dw_dt < tol).any(axis=(-2, -1))

    dw = np.gradient(w, ks, axis=-1)
    d2w = np.gradient(dw, ks, axis=-1)
    g = (1 - ks * dw / (2 * w)) ** 2 - dw**2 / 4 * (1 / w + 0.25) + d2w / 2
    butterfly_violations = (g < tol).any(axis=(-2, -1))

    return cal_violations, butterfly_violations


def check_arbitrage_flat(cfg, iv_flat, tol=-1e-10):
    """check_arbitrage for a flat, ttm-major raveled IV array over cfg's (ttm, k) grid."""
    ttms, ks = grid_from_cfg(cfg)
    iv = iv_flat.reshape(len(ttms), len(ks))
    return check_arbitrage(iv, ttms, ks, tol=tol)