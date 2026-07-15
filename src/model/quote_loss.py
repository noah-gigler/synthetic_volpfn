# Quote loss (no true prices): -log P(bid <= y <= ask) at held-out quote locations
# + calendar/butterfly penalties on a fresh random grid each call, à la
# operator-deep-smoothing-for-implied-volatility's Loss.forward. Expects
# quote_data_preparation batches (query = [random arb grid | held-out quote rows]).

import numpy as np
import torch

from src.data_generation.grid import BUTTERFLY, CAL_LOW, CAL_HIGH


def _n_zb_nrc(cfg):
    z_lim = (cfg["z"]["min"], cfg["z"]["max"])
    rho_lim = (np.sqrt(cfg["ttm"]["min"]), np.sqrt(cfg["ttm"]["max"]))
    n_zb = len(np.arange(z_lim[0], z_lim[1], 0.01))
    n_rc = len(np.arange(rho_lim[0], rho_lim[1], 0.02))
    return n_zb, n_rc


def quote_arb_loss(estimator, batch, logits_BQL, *, cfg, lambda_cal=10.0,
                   lambda_bf=10.0, min_prob=1e-6, return_parts=False):
    # returns per-surface losses (G,) for a possibly grouped batch (G surfaces, E estimators).
    n_zb, n_rc = _n_zb_nrc(cfg)
    BE, Q, _ = logits_BQL.shape
    G = batch.y_query.shape[0]
    E = BE // G
    znorm_bardists = getattr(batch, "znorm_bardists", [batch.znorm_space_bardist] * G)
    raw_bardists = getattr(batch, "raw_bardists", [batch.raw_space_bardist] * G)

    side_cpu = batch.X_query_raw[0, :, 2]
    but_mask_cpu = side_cpu == BUTTERFLY
    lo_mask_cpu, hi_mask_cpu = side_cpu == CAL_LOW, side_cpu == CAL_HIGH
    but_mask, lo_mask, hi_mask = (m.to(logits_BQL.device) for m in (but_mask_cpu, lo_mask_cpu, hi_mask_cpu))
    n_rb = int(but_mask.sum()) // n_zb
    n_zc = int(lo_mask.sum()) // (n_rc - 1)

    z_b = batch.X_query_raw[0, but_mask_cpu, 0].to(logits_BQL.device).reshape(n_rb, n_zb)[0]
    tau_b = batch.X_query_raw[0, but_mask_cpu, 1].to(logits_BQL.device).reshape(n_rb, n_zb)[:, 0]
    r_b = tau_b.sqrt().view(1, n_rb, 1)

    tau_lo = batch.X_query_raw[0, lo_mask_cpu, 1].to(logits_BQL.device).reshape(n_rc - 1, n_zc)[:, 0]
    tau_hi = batch.X_query_raw[0, hi_mask_cpu, 1].to(logits_BQL.device).reshape(n_rc - 1, n_zc)[:, 0]
    r_lo, r_hi = tau_lo.sqrt().view(1, -1, 1), tau_hi.sqrt().view(1, -1, 1)

    losses, nlls, cals, bfs = [], [], [], []
    for g in range(G):
        logits = logits_BQL[g * E:(g + 1) * E]
        bardist = znorm_bardists[g]
        intervals = batch.y_query[g].to(logits_BQL.device)   # (Q, 2) znormed bounds
        mask = torch.isfinite(intervals).all(dim=-1)
        # cdf() asserts on NaN inputs - fill masked-out rows with a valid dummy value
        safe = torch.where(mask[:, None], intervals, bardist.borders[0].expand(Q, 2))
        cdf = bardist.cdf(logits, safe.expand(E, Q, 2))
        p_inside = (cdf[..., 1] - cdf[..., 0]).clamp_min(min_prob)
        nll = -torch.log(p_inside)[:, mask].mean()

        iv = raw_bardists[g].mean(logits).clamp_min(1e-3)
        iv_b = iv[:, but_mask].reshape(E, n_rb, n_zb)
        iv_lo = iv[:, lo_mask].reshape(E, n_rc - 1, n_zc)
        iv_hi = iv[:, hi_mask].reshape(E, n_rc - 1, n_zc)

        # calendar: total variance must increase across maturities at fixed strike k
        # (both slices already share k by construction, see grid.py sample_arb_grid)
        cal = torch.relu(r_lo / r_hi - iv_hi / iv_lo.clamp_min(1e-3)).mean()

        # butterfly: w = iv^2 * tau; at fixed tau d/dk = (1/rho)d/dz -> Gatheral g >= 0
        w = iv_b**2 * tau_b.view(1, n_rb, 1)
        w_z = torch.gradient(w, spacing=(z_b,), dim=-1)[0]
        w_zz = torch.gradient(w_z, spacing=(z_b,), dim=-1)[0]
        w_k, w_kk, k = w_z / r_b, w_zz / r_b**2, z_b.view(1, 1, n_zb) * r_b
        g_fn = (1 - k * w_k / (2 * w)) ** 2 - w_k**2 / 4 * (1 / w + 0.25) + w_kk / 2
        bf = torch.relu(-g_fn).mean()

        losses.append(nll + lambda_cal * cal + lambda_bf * bf)
        nlls.append(nll)
        cals.append(cal)
        bfs.append(bf)

    total = torch.stack(losses)
    if return_parts:
        parts = {"nll": torch.stack(nlls), "cal": torch.stack(cals), "bf": torch.stack(bfs)}
        return total, parts
    return total
