# Quote loss (no true prices): -log P(bid <= y <= ask) at held-out quote locations
# + calendar/butterfly penalties on a fresh random grid each call, à la
# operator-deep-smoothing-for-implied-volatility's Loss.forward. Expects
# quote_data_preparation batches (query = [random arb grid | held-out quote rows]).

import numpy as np
import torch

from src.data_generation.grid import arb_grid_shape


def quote_arb_loss(estimator, batch, logits_BQL, *, cfg, lambda_cal=10.0,
                   lambda_bf=10.0, lambda_reg_z=0.0, lambda_reg_r=0.0,
                   lambda_mean_hinge=0.0, eps_bf=0.0, eps_cal=0.0, min_prob=1e-6,
                   return_parts=False):
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

    losses, nlls, cals, bfs, reg_zs, reg_rs, mean_hinges = [], [], [], [], [], [], []
    ins_fracs, bf_viol_fracs, cal_viol_fracs = [], [], []
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

        iv = raw_bardists[g].mean(logits).clamp_min(1e-3)
        iv_b = iv[:, but_sl].reshape(E, n_rb, n_zb)
        iv_lo = iv[:, lo_sl].reshape(E, n_rc - 1, n_zc)
        iv_hi = iv[:, hi_sl].reshape(E, n_rc - 1, n_zc)

        # calendar: total variance must increase across maturities at fixed strike k
        # (both slices already share k by construction, see grid.py sample_arb_grid)
        # eps_* turn the hinge into a MARGIN (require residual >= eps, not just >= 0). With the
        # plain hinge the gradient vanishes the instant a cell crosses zero, so cells park just
        # inside the boundary and cross back as the fit term pushes the surface around - which
        # is why bf ~ 0.004 still corresponds to several % of violating cells. eps=0 reproduces
        # the old behaviour exactly. NOTE: OpDS's released code uses eps as a TOLERANCE
        # (relu(-g - eps)), the opposite sign; this follows their paper's text instead.
        cal = torch.relu(eps_cal + r_lo / r_hi - iv_hi / iv_lo.clamp_min(1e-3)).mean()

        # butterfly: w = iv^2 * tau; at fixed tau d/dk = (1/rho)d/dz -> Gatheral g >= 0
        w = iv_b**2 * tau_b.view(1, n_rb, 1)
        w_z = torch.gradient(w, spacing=(z_b,), dim=-1)[0]
        w_zz = torch.gradient(w_z, spacing=(z_b,), dim=-1)[0]
        w_k, w_kk, k = w_z / r_b, w_zz / r_b**2, z_b.view(1, 1, n_zb) * r_b
        g_fn = (1 - k * w_k / (2 * w)) ** 2 - w_k**2 / 4 * (1 / w + 0.25) + w_kk / 2
        bf = torch.relu(eps_bf - g_fn).mean()

        # real-time diagnostics (zero weight, not added to the loss sum) - the periodic eval
        # scripts (eval_arbitrage_fine, inside_spread_fraction) only run at the end of a job or
        # on --eval-only; these track the same quantities every epoch, on the same rows already
        # computed above for the hinges, so effectively free. NOT expected to numerically match
        # the eval-time cell_frac exactly - that uses a bigger, finer, jittered grid; these use
        # the smaller training-time arb grid, so read them as a correlated proxy, not the number
        # that goes in a report table.
        with torch.no_grad():
            bf_viol_fracs.append((g_fn < 0).float().mean())
            cal_viol_fracs.append((eps_cal + r_lo / r_hi - iv_hi / iv_lo.clamp_min(1e-3) > 0).float().mean())

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

        # pulls the point estimate (bar-distribution mean) back inside [bid, ask] when it
        # strays outside - the interval NLL only rewards probability MASS between bid/ask, not
        # where the mean sits, so a skewed predictive distribution can satisfy the NLL while its
        # mean is still outside the spread (this is exactly what a low ins%/inside_spread_fraction
        # means). Zero gradient once the mean is inside (a hinge, not a Tikhonov regularizer like
        # reg_z/reg_r - it won't keep pushing after the constraint is met). Excludes is_point rows
        # (regime=0's bid=ask=true) on purpose: those equal the true price, so hinging against them
        # would silently reintroduce a truth-supervised MSE term through the back door.
        n_hi = mask & ~is_point
        # y_query_raw is NaN on arb-grid rows (only held-out rows carry real bid/ask); relu(NaN
        # - iv) is NaN regardless of downstream masking, and that NaN still backprops through
        # the shared forward pass into every query position's logits, corrupting the whole model
        # on the very first step even though the visible mean_hinge value looks finite (the NaN
        # rows are excluded from it via indexing, but NOT from the gradient without this fix) -
        # replace NaN with a safe finite dummy BEFORE the relu, same principle as `safe` above
        y_raw = torch.nan_to_num(batch.y_query_raw[g].to(logits_BQL.device), nan=0.0)

        with torch.no_grad():
            # real-time analogue of eval_real_surfaces' ins% - fraction of held-out (non-point)
            # rows where the predicted mean already lands inside [bid, ask]. Falls back to 0 (not
            # NaN) when n_hi is empty, same convention as mean_hinge below - deliberately avoids
            # reintroducing the empty-selection-mean NaN footgun into a logged/averaged metric.
            # In practice n_hi is never empty now: training's provider never draws regime=0, and
            # regime=0 was excluded from arb val's stratification for the same reason (see
            # ARB_VAL_REGIMES in run_finetuning.py) - this is a defensive fallback, not a live path.
            inside = (iv >= y_raw[:, 0]) & (iv <= y_raw[:, 1])
            ins_fracs.append(inside[:, n_hi].float().mean() if n_hi.any()
                              else torch.zeros((), device=logits_BQL.device))

        if lambda_mean_hinge and n_hi.any():
            below = torch.relu(y_raw[:, 0] - iv)
            above = torch.relu(iv - y_raw[:, 1])
            mean_hinge = (below**2 + above**2)[:, n_hi].mean()
            # n_hi.any() guards a real, not-just-theoretical case: at regime=0 EVERY held-out row
            # is_point (zero-width, bid=ask=true - see add_quote_noise's hard early-return), and
            # _split_by_regime concatenates each regime as a contiguous block before group_size
            # packing groups consecutive same-size surfaces together - so a whole val group can
            # legitimately have zero genuine (nonzero-width) interval rows. Without this guard,
            # `[:, n_hi].mean()` over an empty selection silently returns NaN (a real PyTorch
            # footgun, not an error), which is exactly what made every val_loss slot read NaN
            # despite training looking perfectly healthy - training only avoids it by luck (random
            # per-epoch draws rarely land a whole group in one regime), validation hits it every
            # single time because the frozen val list's regime blocks are deterministic.
        else:
            mean_hinge = torch.zeros((), device=logits_BQL.device)

        losses.append(nll + lambda_cal * cal + lambda_bf * bf
                      + lambda_reg_z * reg_z + lambda_reg_r * reg_r
                      + lambda_mean_hinge * mean_hinge)
        nlls.append(nll)
        cals.append(cal)
        bfs.append(bf)
        reg_zs.append(reg_z)
        reg_rs.append(reg_r)
        mean_hinges.append(mean_hinge)

    total = torch.stack(losses)
    if return_parts:
        parts = {"nll": torch.stack(nlls), "cal": torch.stack(cals), "bf": torch.stack(bfs),
                 "reg_z": torch.stack(reg_zs), "reg_r": torch.stack(reg_rs),
                 "mean_hinge": torch.stack(mean_hinges), "ins_frac": torch.stack(ins_fracs),
                 "bf_viol_frac": torch.stack(bf_viol_fracs), "cal_viol_frac": torch.stack(cal_viol_fracs)}
        return total, parts
    return total
