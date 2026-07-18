# Quote loss (no true prices): -log P(bid <= y <= ask) at held-out quote locations
# + calendar/butterfly penalties on a fresh random grid each call, à la
# operator-deep-smoothing-for-implied-volatility's Loss.forward. Expects
# quote_data_preparation batches (query = [random arb grid | held-out quote rows]).

import numpy as np
import torch

from src.data_generation.grid import arb_grid_shape


def _positive_mean(bardist, logits, eps=1e-6):
    # E[y | y > 0]: mass below 0 is renormalized away instead of clamping the mean,
    # so there's no flat region for the finite-difference penalties to blow up on.
    # Buckets straddling 0 contribute their positive part (uniform-in-bucket, same
    # approximation cdf() uses for the half-normal tails); right tail keeps the
    # half-normal mean to match FullSupportBarDistribution.mean.
    borders, widths = bardist.borders, bardist.bucket_widths
    means = (borders[:-1].clamp_min(0.0) + borders[1:].clamp_min(0.0)) / 2
    means = means.clone()
    means[-1] = borders[-2] + bardist.halfnormal_with_p_weight_before(widths[-1]).mean
    share_pos = (borders[1:] / widths).clamp(0.0, 1.0)  # fraction of bucket above 0
    p = logits.softmax(-1) * share_pos
    return (p @ means) / p.sum(-1).clamp_min(eps)


def quote_arb_loss(estimator, batch, logits_BQL, *, cfg, lambda_cal=10.0,
                   lambda_bf=10.0, lambda_neg=1.0, lambda_reg_z=0.0,
                   lambda_reg_r=0.0, min_prob=1e-6, return_parts=False):
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

    losses, nlls, cals, bfs, negs, reg_zs, reg_rs = [], [], [], [], [], [], []
    for g in range(G):
        logits = logits_BQL[g * E:(g + 1) * E]
        bardist = znorm_bardists[g]
        intervals = batch.y_query[g].to(logits_BQL.device)   # (Q, 2) znormed bounds
        mask = torch.isfinite(intervals).all(dim=-1)
        # a zero-width interval (bid==ask, e.g. regime=0's exact quotes) makes
        # cdf(ask)-cdf(bid) exactly 0 for ANY prediction - not just bad ones, since a
        # continuous distribution always assigns zero mass to a single point. That silently
        # floors every such row to the same constant -log(min_prob) with no gradient signal
        # toward the true value. Use the point density NLL (bardist.forward, the same
        # bucket-cross-entropy the plain supervised path uses) for those rows instead.
        is_point = mask & (intervals[:, 0] == intervals[:, 1])

        # cdf()/forward() assert on NaN/out-of-range inputs - fill masked-out rows with a
        # valid dummy; bid doubles as the point target since bid==ask on point rows anyway
        safe = torch.where(mask[:, None], intervals, bardist.borders[0].expand(Q, 2))
        cdf = bardist.cdf(logits, safe.expand(E, Q, 2))
        interval_nll = -torch.log((cdf[..., 1] - cdf[..., 0]).clamp_min(min_prob))
        # bardist.forward has no floor on the target bucket's probability (unlike the interval
        # path's min_prob clamp above), so it's unbounded above if the model transiently puts
        # ~0 probability on the true bucket - clamp to the same ceiling for stability/symmetry
        point_nll = bardist.forward(logits, safe[:, 0].expand(E, Q)).clamp_max(-np.log(min_prob))

        nll = torch.where(is_point.expand(E, Q), point_nll, interval_nll)[:, mask].mean()

        # root-cause penalty for negative-IV mass: the interval NLL never sees mass below 0
        # (only mass inside [bid, ask]), so unquoted extrapolation cells are otherwise free
        # to put weight on negative buckets, dragging mean() negative. raw-space cdf at 0 is
        # P(IV <= 0) directly (affine border transform preserves bucket probabilities);
        # inert (exactly 0) on surfaces whose raw borders start above 0.
        zero = torch.zeros(E, Q, 1, device=logits.device, dtype=logits.dtype)
        p_neg = raw_bardists[g].cdf(logits, zero)[..., 0]
        neg = -torch.log((1 - p_neg).clamp_min(min_prob)).mean()

        iv = _positive_mean(raw_bardists[g], logits).clamp_min(1e-3)
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

        losses.append(nll + lambda_cal * cal + lambda_bf * bf + lambda_neg * neg
                      + lambda_reg_z * reg_z + lambda_reg_r * reg_r)
        nlls.append(nll)
        cals.append(cal)
        bfs.append(bf)
        negs.append(neg)
        reg_zs.append(reg_z)
        reg_rs.append(reg_r)

    total = torch.stack(losses)
    if return_parts:
        parts = {"nll": torch.stack(nlls), "cal": torch.stack(cals), "bf": torch.stack(bfs),
                 "neg": torch.stack(negs), "reg_z": torch.stack(reg_zs), "reg_r": torch.stack(reg_rs)}
        return total, parts
    return total
