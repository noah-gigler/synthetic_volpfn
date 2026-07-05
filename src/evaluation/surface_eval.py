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


