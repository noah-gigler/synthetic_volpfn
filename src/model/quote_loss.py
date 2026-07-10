# Quote loss (no true prices): -log P(bid <= y <= ask) at held-out quote locations
# + calendar/butterfly hinges on the mean surface. Expects quote_data_preparation batches.

import torch


def quote_arb_loss(estimator, batch, logits_BQL, *, grid_shape, lambda_cal=1.0,
                   lambda_bf=1.0, min_prob=1e-6):
    # returns per-surface losses (G,) for a possibly grouped batch (G surfaces, E estimators).
    # query = [arb lattice (first n_grid rows) | held-out quote rows]; arb reads the lattice,
    # NLL reads the finite (held-out) rows.
    BE, Q, _ = logits_BQL.shape
    n_ttm, n_z = grid_shape
    n_grid = n_ttm * n_z
    assert Q >= n_grid, "query must hold the full arb lattice as its first n_grid rows"

    G = batch.y_query.shape[0]
    E = BE // G
    znorm_bardists = getattr(batch, "znorm_bardists", [batch.znorm_space_bardist] * G)
    raw_bardists = getattr(batch, "raw_bardists", [batch.raw_space_bardist] * G)

    # (ρ, z) recovered from the lattice rows; feature col 0 is z directly (shared across ttm rows)
    tau = batch.X_query_raw[0, :n_grid, 1].to(logits_BQL.device).reshape(n_ttm, n_z)
    zz = batch.X_query_raw[0, :n_grid, 0].to(logits_BQL.device).reshape(n_ttm, n_z)
    rho = torch.sqrt(tau[:, 0])                          # (n_ttm,)
    zs = zz[0]                                           # (n_z,)
    r_, z_ = rho.view(1, n_ttm, 1), zs.view(1, 1, n_z)

    losses = []
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

        # arbitrage hinges on the lattice rows, in (ρ, z) with w = iv^2 * tau
        iv = raw_bardists[g].mean(logits[:, :n_grid]).clamp_min(1e-3).reshape(E, n_ttm, n_z)
        w = iv**2 * tau

        # calendar: dw/dtau|_k = (1/2ρ)[w_ρ - (z/ρ) w_z] >= 0   (ρ uniform -> clean stencil)
        w_rho = torch.gradient(w, spacing=(rho,), dim=-2)[0]
        w_z = torch.gradient(w, spacing=(zs,), dim=-1)[0]
        cal = torch.relu(-(w_rho - (z_ / r_) * w_z) / (2 * r_)).mean()

        # butterfly: at fixed tau d/dk = (1/ρ)d/dz -> w_k=w_z/ρ, w_kk=w_zz/ρ^2, k=zρ; Gatheral g >= 0
        w_zz = torch.gradient(w_z, spacing=(zs,), dim=-1)[0]
        w_k, w_kk, k = w_z / r_, w_zz / r_**2, z_ * r_
        g_fn = (1 - k * w_k / (2 * w)) ** 2 - w_k**2 / 4 * (1 / w + 0.25) + w_kk / 2
        bf = torch.relu(-g_fn).mean()

        losses.append(nll + lambda_cal * cal + lambda_bf * bf)

    return torch.stack(losses)
