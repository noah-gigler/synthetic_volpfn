import numpy as np
import torch

from tabpfn import TabPFNRegressor
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.constants import ModelVersion

from src.data_generation.grid import Grid, arb_grid_shape
from src.model.preprocessed_dataset import preprocess_surfaces

_PERF = PerformanceOptions(force_recompute_layer=False, use_chunkwise_inference=False)
_eval_estimators: dict = {}  # model_version -> (estimator, pretrained_state)


def _get_eval_estimator(model_version=None):
    # one batched estimator per version, built once and reused; callers swap weights via
    # load_state_dict instead of re-fitting/reloading per surface (the old per-surface path
    # was the bottleneck).
    global _eval_estimators
    if model_version not in _eval_estimators:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        common = dict(
            fit_mode="batched", n_estimators=1, device=device,
            categorical_features_indices=[],
            inference_config={
                "FINGERPRINT_FEATURE": False,
                "FEATURE_SHIFT_METHOD": None,
                "PREPROCESS_TRANSFORMS": [PreprocessorConfig("none", categorical_name="numeric")],
            },
        )
        if model_version is not None:
            est = TabPFNRegressor.create_default_for_version(version=ModelVersion(model_version), **common)
        else:
            est = TabPFNRegressor(**common)
        est._initialize_model_variables()
        est.model_.to(device)
        est.model_.eval()
        pretrained_state = {k: v.detach().cpu().clone() for k, v in est.model_.state_dict().items()}
        _eval_estimators[model_version] = (est, pretrained_state)
    return _eval_estimators[model_version]


def _forward_logits(est, surfaces):
    # yields (raw_bardist, logits (E,Q,L), y_query_raw (Q,)) per surface, in input order;
    # shared by every eval metric that needs the predictive distribution, not just its mean.
    # torch.no_grad() wraps the loop body, not the function - a decorator on a generator
    # function only guards the call that creates the generator, not each lazily-run `next()`
    with torch.no_grad():
        for s in surfaces:
            est.raw_space_bardist_ = s.raw_space_bardist
            est.znorm_space_bardist_ = s.znorm_space_bardist
            est.fit_from_preprocessed(
                s.X_context, s.y_context, s.cat_indices, s.configs,
                performance_options=_PERF, no_refit=True,
            )
            _, per_estim_logits, _ = est.forward(s.X_query, use_inference_mode=False)
            logits_QBEL = torch.stack(per_estim_logits, dim=2)
            Q, B, E, L = logits_QBEL.shape
            logits_BQL = logits_QBEL.permute(1, 2, 0, 3).reshape(B * E, Q, L)
            for g in range(B):
                yield s.raw_bardists[g], logits_BQL[g * E:(g + 1) * E], s.y_query_raw[g]


def _predict_raw(est, surfaces):
    # raw-space IV mean per surface, in the input surface order (groups preserve order)
    preds = []
    for bardist, logits_EQL, y_query_raw in _forward_logits(est, surfaces):
        # bardist may live on the model's device (e.g. CUDA); keep the mean computation
        # there and only move the final per-surface result to CPU for numpy conversion
        iv = bardist.mean(logits_EQL).mean(0).cpu()  # (Q,)
        preds.append((iv.numpy(), y_query_raw.numpy()))
    return preds


def check_arbitrage(iv, ttms, zs, tol=-1e-10):
    # (ρ, z) coordinates: ρ=√τ, z=k/√τ, k=z·√τ; total variance w = iv^2 * τ
    w = iv**2 * ttms[:, None]
    rho = np.sqrt(ttms)
    r_, z_ = rho[:, None], zs[None, :]

    # calendar: dw/dtau|_k = (1/2ρ)[w_ρ - (z/ρ) w_z] >= 0
    w_rho = np.gradient(w, rho, axis=-2)
    w_z = np.gradient(w, zs, axis=-1)
    cal = (w_rho - (z_ / r_) * w_z) / (2 * r_)
    cal_violations = (cal < tol).any(axis=(-2, -1))

    # butterfly: w_k=w_z/ρ, w_kk=w_zz/ρ^2, k=zρ; Gatheral g >= 0
    w_zz = np.gradient(w_z, zs, axis=-1)
    w_k, w_kk, k = w_z / r_, w_zz / r_**2, z_ * r_
    g = (1 - k * w_k / (2 * w)) ** 2 - w_k**2 / 4 * (1 / w + 0.25) + w_kk / 2
    butterfly_violations = (g < tol).any(axis=(-2, -1))

    return cal_violations, butterfly_violations


def check_arbitrage_flat(cfg, iv_flat, tol=-1e-10):
    g = Grid(cfg)
    return check_arbitrage(iv_flat.reshape(g.shape), g.ttms, g.zs, tol=tol)


def eval_surfaces(model, train_list, test_list, cfg, reload_state=None, group_size=128, model_version=None,
                   iv_max=1.5, y_mean=0.300, y_scale=0.135, feature_scale=None):
    # `model` is ignored except as an API anchor; a shared batched estimator does the work.
    # reload_state=None evaluates the non-finetuned pretrained weights.
    est, pretrained_state = _get_eval_estimator(model_version)
    est.model_.load_state_dict(reload_state if reload_state is not None else pretrained_state)

    rng = np.random.default_rng(0)
    # cap the forward batch so peak GPU memory stays bounded regardless of len(train_list);
    # ~16 same-shape surfaces peaks ~5 GiB, well under a 14.5 GiB card (scripts/probe_eval_batch.py)
    surfaces = preprocess_surfaces(est, train_list, test_list, rng, iv_max, group_size=group_size,
                                     y_mean=y_mean, y_scale=y_scale, feature_scale=feature_scale)

    maes, mapes, cal_violations, butterfly_violations = [], [], [], []
    for y_pred, y_te in _predict_raw(est, surfaces):
        maes.append(np.mean(np.abs(y_te - y_pred)))
        mapes.append(np.mean(np.abs((y_te - y_pred) / y_te)) * 100)
        cal_v, butterfly_v = check_arbitrage_flat(cfg, y_pred)
        cal_violations.append(cal_v)
        butterfly_violations.append(butterfly_v)

    return tuple(np.mean(x) for x in (maes, mapes, cal_violations, butterfly_violations))


def quantile_coverage(model, train_list, test_list, reload_state=None, levels=(0.2, 0.5, 0.8)):
    """Empirical coverage of central predictive intervals on the query points.
    Returns {level: mean fraction of true values inside the central `level` interval}."""
    qs = sorted({q for lv in levels for q in ((1 - lv) / 2, 1 - (1 - lv) / 2)})
    coverages = {lv: [] for lv in levels}
    for (X_tr, y_tr), (X_te, y_te) in zip(train_list, test_list):
        model.fit(X_tr, y_tr)
        if reload_state is not None:
            model.model_.load_state_dict(reload_state)
        preds = model.predict(X_te, output_type="quantiles", quantiles=list(qs))
        for lv in levels:
            lo = preds[qs.index((1 - lv) / 2)]
            hi = preds[qs.index(1 - (1 - lv) / 2)]
            coverages[lv].append(np.mean((y_te >= lo) & (y_te <= hi)))
    return {lv: float(np.mean(c)) for lv, c in coverages.items()}


def eval_uncertainty(model, train_list, test_list, reload_state=None, group_size=128, model_version=None,
                      iv_max=1.5, y_mean=0.300, y_scale=0.135, feature_scale=None,
                      pinball_levels=(0.05, 0.95), interval_level=0.90, n_crps_quantiles=39):
    """Predictive-distribution calibration: CRPS, pinball loss, central-interval width, and PIT
    values, all read off the bar distribution's own `cdf`/`icdf` - no extra model output needed,
    same batched forward pass as `eval_surfaces`. CRPS has no closed form for a piecewise-uniform
    bar distribution, so it's estimated via the standard quantile-average identity
    `CRPS(F,y) = 2 * E_tau~U(0,1)[pinball_tau(y, F^-1(tau))]` (Gneiting & Raftery 2007) on a
    `n_crps_quantiles`-point grid. PIT values should be ~Uniform(0,1) under good calibration -
    return the raw array so callers can plot a PIT histogram.
    """
    est, pretrained_state = _get_eval_estimator(model_version)
    est.model_.load_state_dict(reload_state if reload_state is not None else pretrained_state)

    rng = np.random.default_rng(0)
    surfaces = preprocess_surfaces(est, train_list, test_list, rng, iv_max, group_size=group_size,
                                     y_mean=y_mean, y_scale=y_scale, feature_scale=feature_scale)

    tau_grid = (np.arange(n_crps_quantiles) + 0.5) / n_crps_quantiles  # midpoint rule, avoids tau=0,1
    lo_level, hi_level = (1 - interval_level) / 2, 1 - (1 - interval_level) / 2

    crps_all, pit_all = [], []
    pinball_all = {lv: [] for lv in pinball_levels}
    widths_all = []
    for bardist, logits_EQL, y_query_raw in _forward_logits(est, surfaces):
        y = y_query_raw.to(logits_EQL.device)  # (Q,)

        def icdf_ens(level):
            # average the per-estimator quantile prediction across the ensemble (E=1 in
            # practice, since finetune.py fixes n_estimators=1 - see CLAUDE.md)
            return torch.stack([bardist.icdf(logits_EQL[e], level) for e in range(logits_EQL.shape[0])], 0).mean(0)

        def pinball(level, q_pred):
            diff = y - q_pred
            return torch.maximum(level * diff, (level - 1) * diff)

        # bardist.cdf's 1-D-ys branch broadcasts ys as points *shared* across the batch dim,
        # not one-per-row - our y is one-per-row and happens to share Q's length, so an
        # unsqueezed trailing dim is needed to hit its positional (asserted-shape) branch instead
        pit = torch.stack(
            [bardist.cdf(logits_EQL[e], y.unsqueeze(-1)).squeeze(-1) for e in range(logits_EQL.shape[0])], 0,
        ).mean(0)
        pit_all.append(pit.cpu().numpy())

        lo, hi = icdf_ens(lo_level), icdf_ens(hi_level)
        widths_all.append((hi - lo).cpu().numpy())

        for lv in pinball_levels:
            pinball_all[lv].append(pinball(lv, icdf_ens(lv)).cpu().numpy())

        crps_terms = torch.stack([pinball(tau, icdf_ens(tau)) for tau in tau_grid], 0)  # (n_crps_quantiles, Q)
        crps_all.append((2 * crps_terms.mean(0)).cpu().numpy())

    return dict(
        crps=float(np.mean(np.concatenate(crps_all))),
        pinball={lv: float(np.mean(np.concatenate(v))) for lv, v in pinball_all.items()},
        mean_interval_width=float(np.mean(np.concatenate(widths_all))),
        pit=np.concatenate(pit_all),
    )


def eval_arbitrage_fine(model, train_list, cfg, arb_rows, reload_state=None, group_size=8, model_version=None,
                         iv_max=1.5, y_mean=0.300, y_scale=0.135, feature_scale=None):
    """Cell-level arb diagnostics on a single frozen arb grid, shared across every surface in
    `train_list` and reused as-is across eval runs/checkpoints (produced once via
    `sample_arb_grid`, see `run_finetuning.load_arb_grid`) - unlike training's grid, this one is
    NOT re-sampled per surface, so every surface shares the same query shape and can be batched
    the same way `eval_surfaces` batches the MAE pass.

    Sign convention: metric < 0 means violated (mirrors Gatheral's g >= 0 no-arb condition), so
    across both butterfly and calendar cells "more negative" always means "worse". Expects the
    bid/ask-free schema from `quote_data_preparation`/`noisy_data_preparation` (`X = [z, tau, side]`).
    `group_size` defaults far lower than `eval_surfaces`'s: the arb grid is ~10x more rows per
    surface than the native MAE grid, so peak memory grows much faster with batch size (see
    scripts/bench_arb_val_batch.py - group_size=32 already hits ~84GB, 64 OOMs on a 96GB card).
    """
    est, pretrained_state = _get_eval_estimator(model_version)
    est.model_.load_state_dict(reload_state if reload_state is not None else pretrained_state)

    query = arb_rows.copy()
    query[:, 2] = 0.0
    test_list = [(query, np.zeros(len(query)))] * len(train_list)  # y unused, pred-only query
    rng = np.random.default_rng(0)
    surfaces = preprocess_surfaces(est, train_list, test_list, rng, iv_max, group_size=group_size,
                                     y_mean=y_mean, y_scale=y_scale, feature_scale=feature_scale)

    n_zb, n_rc, n_zc = arb_grid_shape(cfg)
    n_cal = (n_rc - 1) * n_zc
    n_but = len(arb_rows) - 2 * n_cal
    n_rb = n_but // n_zb
    but_sl = slice(0, n_but)
    lo_sl = slice(n_but, n_but + n_cal)
    hi_sl = slice(n_but + n_cal, n_but + 2 * n_cal)

    z_b = arb_rows[but_sl, 0].reshape(n_rb, n_zb)[0]
    tau_b = arb_rows[but_sl, 1].reshape(n_rb, n_zb)[:, 0]
    r_b = np.sqrt(tau_b)[:, None]

    tau_lo = arb_rows[lo_sl, 1].reshape(n_rc - 1, n_zc)[:, 0]
    tau_hi = arb_rows[hi_sl, 1].reshape(n_rc - 1, n_zc)[:, 0]
    r_lo, r_hi = np.sqrt(tau_lo)[:, None], np.sqrt(tau_hi)[:, None]

    cell_fracs, mean_depths, worst_cells, arb_free = [], [], [], []
    for pred, _ in _predict_raw(est, surfaces):
        iv_b = np.maximum(pred[but_sl].reshape(n_rb, n_zb), 1e-3)
        w = iv_b**2 * tau_b[:, None]
        w_z = np.gradient(w, z_b, axis=-1)
        w_zz = np.gradient(w_z, z_b, axis=-1)
        w_k, w_kk, k = w_z / r_b, w_zz / r_b**2, z_b[None, :] * r_b
        bf_metric = (1 - k * w_k / (2 * w)) ** 2 - w_k**2 / 4 * (1 / w + 0.25) + w_kk / 2

        iv_lo = np.maximum(pred[lo_sl].reshape(n_rc - 1, n_zc), 1e-3)
        iv_hi = pred[hi_sl].reshape(n_rc - 1, n_zc)
        cal_metric = iv_hi / iv_lo - r_lo / r_hi

        metric = np.concatenate([bf_metric.ravel(), cal_metric.ravel()])
        viol = metric < 0
        cell_fracs.append(viol.mean())
        mean_depths.append(metric[viol].mean() if viol.any() else 0.0)
        worst_cells.append(metric.min())
        arb_free.append(0.0 if viol.any() else 1.0)

    return tuple(float(np.mean(x)) for x in (cell_fracs, mean_depths, worst_cells, arb_free))


def eval_real_surfaces(model, train_list, test_list, reload_state=None, group_size=64, model_version=None,
                        iv_max=1.5, y_mean=0.300, y_scale=0.135, feature_scale=None):
    """Truth-free scoring for real quotes: MAE vs mid and inside-[bid,ask] fraction, at the
    held-out quote locations and at the context quotes. Expects `make_real_eval_set(..., cfg=None)`
    pairs: context X = [z, tau, side] (bids then asks), test = (held_rows, y_held (n, 2) [bid, ask])."""
    est, pretrained_state = _get_eval_estimator(model_version)
    est.model_.load_state_dict(reload_state if reload_state is not None else pretrained_state)

    queries, targets = [], []
    for (X_tr, y_tr), (X_held, y_held) in zip(train_list, test_list):
        nc = len(y_tr) // 2
        X_ctx = X_tr[:nc].copy()
        X_ctx[:, 2] = 0.0
        Xq = np.vstack([X_ctx, X_held])
        bid = np.concatenate([y_tr[:nc], y_held[:, 0]])
        ask = np.concatenate([y_tr[nc:], y_held[:, 1]])
        queries.append((Xq, np.zeros(len(Xq))))  # y unused, pred-only query
        targets.append((nc, bid, ask))

    rng = np.random.default_rng(0)
    surfaces = preprocess_surfaces(est, train_list, queries, rng, iv_max, group_size=group_size,
                                     y_mean=y_mean, y_scale=y_scale, feature_scale=feature_scale)

    out = {k: [] for k in ("mae_held", "inside_held", "mae_ctx", "inside_ctx")}
    for (pred, _), (nc, bid, ask) in zip(_predict_raw(est, surfaces), targets):
        mid = (bid + ask) / 2
        inside = (pred >= bid) & (pred <= ask)
        err = np.abs(pred - mid)
        out["mae_held"].append(err[nc:].mean())
        out["inside_held"].append(inside[nc:].mean())
        out["mae_ctx"].append(err[:nc].mean())
        out["inside_ctx"].append(inside[:nc].mean())
    return {k: float(np.mean(v)) for k, v in out.items()}


def inside_spread_fraction(model, train_list, reload_state=None):
    """Fraction of predictions at the quote locations lying within [bid, ask].
    Expects the bid/ask schema from src.data_generation.noise: X = [k, tau, side]
    with the first half of the rows bids and the second half asks."""
    fracs = []
    for X_tr, y_tr in train_list:
        model.fit(X_tr, y_tr)
        if reload_state is not None:
            model.model_.load_state_dict(reload_state)
        n = len(y_tr) // 2
        X_query = X_tr[:n].copy()
        X_query[:, 2] = 0.0  # ask for the true value at the quoted (k, tau)
        y_pred = model.predict(X_query)
        bid, ask = y_tr[:n], y_tr[n:]
        fracs.append(np.mean((y_pred >= bid) & (y_pred <= ask)))
    return float(np.mean(fracs))


