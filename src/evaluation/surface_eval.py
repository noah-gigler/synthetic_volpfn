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
    ttms, ks = grid_from_cfg(cfg)
    iv = iv_flat.reshape(len(ttms), len(ks))
    return check_arbitrage(iv, ttms, ks, tol=tol)


def eval_surfaces(model, train_list, test_list, cfg, reload_state=None):
    maes, mapes, cal_violations, butterfly_violations = [], [], [], []
    for (X_tr, y_tr), (X_te, y_te) in zip(train_list, test_list):
        model.fit(X_tr, y_tr) # always resets weights (but is still needed to preprocess data)
        if reload_state is not None:
            model.model_.load_state_dict(reload_state)  # restore weights if finetuned
        y_pred = model.predict(X_te)

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


