import numpy as np
from src.data_generation.SSVI import ssvi, sample_params

# gaussian k sampling with mean ATM
# uniform ttm sampling as bins are already exponentially distributed
def sample_sparse_points(ks, ttms, n_points, n_samples):
    k_weights = np.exp(-0.5 * (ks / 0.25) ** 2)            # +-0.4 at 5th-95th pct
    k_weights /= k_weights.sum()

    flat_weights = np.tile(k_weights, len(ttms))
    flat_weights /= flat_weights.sum()

    scalar = np.isscalar(n_points)
    n_points = np.broadcast_to(np.asarray(n_points, dtype=int), (n_samples,))

    flat_idx = [
        np.random.choice(len(ks) * len(ttms), size=m, replace=False, p=flat_weights)
        for m in n_points
    ]

    if scalar:
        t_idx, k_idx = np.unravel_index(np.array(flat_idx), (len(ttms), len(ks)))
        return k_idx, t_idx

    pairs = [np.unravel_index(fi, (len(ttms), len(ks))) for fi in flat_idx]
    return [p[1] for p in pairs], [p[0] for p in pairs]


def sample_context_sizes(n_context, n, dist="uniform"):
    if np.isscalar(n_context):
        return np.full(n, n_context, dtype=int)
    lo, hi = n_context
    if dist == "uniform":
        return np.random.randint(lo, hi + 1, size=n)
    u = np.random.uniform(np.log(lo), np.log(hi + 1), size=n)
    return np.minimum(np.exp(u).astype(int), hi)


def grid_from_cfg(cfg):
    ttms = np.geomspace(cfg["ttm"]["min"], cfg["ttm"]["max"], cfg["ttm"]["n_points"])
    ks = np.linspace(cfg["k"]["min"], cfg["k"]["max"], cfg["k"]["n_points"])
    return ttms, ks


def generate_surfaces(cfg, n):
    ttms, ks = grid_from_cfg(cfg)

    rho, eta, gamma, v_bar, v0, kappa = sample_params(cfg, n)
    surfaces = ssvi(ttms, ks, rho, eta, gamma, v_bar, v0, kappa) 

    return ttms, ks, surfaces


def _split_context_query(ks, ttms, surfaces, k_idx, t_idx):
    TT, KK = np.meshgrid(ttms, ks, indexing='ij')
    k_flat   = KK.ravel()
    tau_flat = TT.ravel()

    train, test = [], []
    for i in range(len(surfaces)):
        sigma = surfaces[i].ravel()
        train_idx = t_idx[i] * len(ks) + k_idx[i]

        # query is the full grid, including context points
        X_train, y_train = np.column_stack([k_flat[train_idx], tau_flat[train_idx]]), sigma[train_idx]
        X_test,  y_test  = np.column_stack([k_flat, tau_flat]), sigma

        train.append((X_train, y_train))
        test.append((X_test,  y_test))

    return train, test


def data_preparation(cfg, n, n_context, size_dist="uniform"):
    ttms, ks, surfaces = generate_surfaces(cfg, n)
    sizes = sample_context_sizes(n_context, n, dist=size_dist)
    k_idx, t_idx = sample_sparse_points(ks, ttms, sizes, n_samples=n)
    return _split_context_query(ks, ttms, surfaces, k_idx, t_idx)


def make_stratified_eval_set(cfg, n_surfaces, context_sizes):
    ttms, ks, surfaces = generate_surfaces(cfg, n_surfaces)

    train, test = [], []
    for size in context_sizes:
        k_idx, t_idx = sample_sparse_points(ks, ttms, np.full(n_surfaces, size), n_samples=n_surfaces)
        tr, te = _split_context_query(ks, ttms, surfaces, k_idx, t_idx)
        train += tr
        test += te

    return train, test


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))
    train, test = data_preparation(cfg, 1, 20)
    from tabpfn import TabPFNRegressor

    X_train, y_train = train[0]
    X_test,  y_test  = test[0]

    model = TabPFNRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)