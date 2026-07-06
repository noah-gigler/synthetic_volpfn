# Quote loss (no true prices): -log P(bid <= y <= ask) at held-out quote locations
# + calendar/butterfly hinges on the mean surface. Expects quote_data_preparation batches.

import torch


def quote_arb_loss(estimator, batch, logits_BQL, *, grid_shape, lambda_cal=1.0,
                   lambda_bf=1.0, min_prob=1e-6):
    BE, Q, _ = logits_BQL.shape
    n_ttm, n_k = grid_shape
    assert Q == n_ttm * n_k, "query must be the full ttm-major grid"

    bardist = batch.znorm_space_bardist
    intervals = batch.y_query[0].to(logits_BQL.device)   # (Q, 2) znormed bounds
    mask = torch.isfinite(intervals).all(dim=-1)
    # cdf() asserts on NaN inputs - fill masked-out rows with a valid dummy value
    safe = torch.where(mask[:, None], intervals, bardist.borders[0].expand(Q, 2))
    cdf = bardist.cdf(logits_BQL, safe.expand(BE, Q, 2))
    p_inside = (cdf[..., 1] - cdf[..., 0]).clamp_min(min_prob)
    nll = -torch.log(p_inside)[:, mask].mean()

    # arbitrage hinges in total-variance space (w = iv^2 * tau) on the raw-space mean
    iv = batch.raw_space_bardist.mean(logits_BQL).clamp_min(1e-3).reshape(BE, n_ttm, n_k)
    tau = batch.X_query_raw[0, :, 1].to(logits_BQL.device).reshape(n_ttm, n_k)
    ks = batch.X_query_raw[0, :n_k, 0].to(logits_BQL.device)
    w = iv**2 * tau

    cal = torch.relu(-(w[:, 1:, :] - w[:, :-1, :])).mean()

    dw = torch.gradient(w, spacing=(ks,), dim=-1)[0]
    d2w = torch.gradient(dw, spacing=(ks,), dim=-1)[0]
    g = (1 - ks * dw / (2 * w)) ** 2 - dw**2 / 4 * (1 / w + 0.25) + d2w / 2
    bf = torch.relu(-g).mean()

    return nll + lambda_cal * cal + lambda_bf * bf
