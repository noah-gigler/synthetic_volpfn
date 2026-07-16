# Quote loss (no true prices): -log P(bid <= y <= ask) at held-out quote locations
# + calendar/butterfly penalties on a fresh random grid each call, à la
# operator-deep-smoothing-for-implied-volatility's Loss.forward. Expects
# quote_data_preparation batches (query = [random arb grid | held-out quote rows]).

import numpy as np
import torch

from src.data_generation.grid import arb_grid_shape


def quote_arb_loss(estimator, batch, logits_BQL, *, cfg, lambda_cal=10.0,
                   lambda_bf=10.0, lambda_reg_z=0.0, lambda_reg_r=0.0,
                   min_prob=1e-6, return_parts=False):
    # returns per-surface losses (G,) for a possibly grouped batch (G surfaces, E estimators).
    # row layout is fixed and positional (see grid.py's sample_arb_grid/arb_grid_shape docstring):
    # query = [n_rb*n_zb butterfly rows | (n_rc-1)*n_zc cal_lo rows | (n_rc-1)*n_zc cal_hi rows | held-out rows]
    n_zb, n_rc, n_zc = arb_grid_shape(cfg)
    BE, Q, _ = logits_BQL.shape
    G = batch.y_query.shape[0]
    E = BE // G
    znorm_bardists = getattr(batch, "znorm_bardists", [batch.znorm_space_bardist] * G)
    raw_bardists = getattr(batch, "raw_bardists", [batch.raw_space_bardist] * G)

    n_cal = (n_rc - 1) * n_zc
    intervals0 = batch.y_query[0]
    n_heldout = int(torch.isfinite(intervals0).all(dim=-1).sum())
    n_but = Q - n_heldout - 2 * n_cal
    n_rb = n_but // n_zb

    but_sl = slice(0, n_but)
    lo_sl = slice(n_but, n_but + n_cal)
    hi_sl = slice(n_but + n_cal, n_but + 2 * n_cal)

    z_b = batch.X_query_raw[0, but_sl, 0].to(logits_BQL.device).reshape(n_rb, n_zb)[0]
    tau_b = batch.X_query_raw[0, but_sl, 1].to(logits_BQL.device).reshape(n_rb, n_zb)[:, 0]
    r_b = tau_b.sqrt().view(1, n_rb, 1)

    tau_lo = batch.X_query_raw[0, lo_sl, 1].to(logits_BQL.device).reshape(n_rc - 1, n_zc)[:, 0]
    tau_hi = batch.X_query_raw[0, hi_sl, 1].to(logits_BQL.device).reshape(n_rc - 1, n_zc)[:, 0]
    r_lo, r_hi = tau_lo.sqrt().view(1, -1, 1), tau_hi.sqrt().view(1, -1, 1)

    losses, nlls, cals, bfs, reg_zs, reg_rs = [], [], [], [], [], []
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
        iv_b = iv[:, but_sl].reshape(E, n_rb, n_zb)
        iv_lo = iv[:, lo_sl].reshape(E, n_rc - 1, n_zc)
        iv_hi = iv[:, hi_sl].reshape(E, n_rc - 1, n_zc)

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

        # curvature regularizers (à la OpDS Loss.reg_z/reg_r): raw-IV roughness on the
        # butterfly grid, independent of whether a hinge actually fires - the mechanism
        # that lets a dense-but-fixed z_b (and sparse r_b) stay safe between grid points
        if lambda_reg_z or lambda_reg_r:
            iv_z = torch.gradient(iv_b, spacing=(z_b,), dim=-1)[0]
            iv_zz = torch.gradient(iv_z, spacing=(z_b,), dim=-1)[0]
            reg_z = iv_zz.square().mean().sqrt()

            r_b_1d = r_b.view(-1)
            iv_r = torch.gradient(iv_b, spacing=(r_b_1d,), dim=-2)[0]
            iv_rr = torch.gradient(iv_r, spacing=(r_b_1d,), dim=-2)[0]
            reg_r = iv_rr.square().mean().sqrt()
        else:
            reg_z = torch.zeros((), device=logits_BQL.device)
            reg_r = torch.zeros((), device=logits_BQL.device)

        losses.append(nll + lambda_cal * cal + lambda_bf * bf + lambda_reg_z * reg_z + lambda_reg_r * reg_r)
        nlls.append(nll)
        cals.append(cal)
        bfs.append(bf)
        reg_zs.append(reg_z)
        reg_rs.append(reg_r)

    total = torch.stack(losses)
    if return_parts:
        parts = {"nll": torch.stack(nlls), "cal": torch.stack(cals), "bf": torch.stack(bfs),
                 "reg_z": torch.stack(reg_zs), "reg_r": torch.stack(reg_rs)}
        return total, parts
    return total
