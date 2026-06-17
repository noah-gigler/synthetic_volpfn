import numpy as np
from src.data_generation.SSVI import ssvi, sample_params

# gaussian k sampling with mean ATM
# uniform ttm sampling as bins are already exponentially distributed
def sample_sparse_points(ks, ttms, n_points, n_samples):
    k_weights = np.exp(-0.5 * (ks / 0.25) ** 2)            # +-0.4 at 5th-95th pct
    k_weights /= k_weights.sum()

    flat_weights = np.tile(k_weights, len(ttms))
    flat_weights /= flat_weights.sum()

    flat_idx = np.array([
        np.random.choice(len(ks) * len(ttms), size=n_points, replace=False, p=flat_weights)
        for _ in range(n_samples)
    ])

    t_idx, k_idx = np.unravel_index(flat_idx, (len(ttms), len(ks)))
    return k_idx, t_idx


def generate_surfaces(cfg, n):
    ttms = np.geomspace(cfg["ttm"]["min"], cfg["ttm"]["max"], cfg["ttm"]["n_points"])
    ks = np.linspace(cfg["k"]["min"], cfg["k"]["max"], cfg["k"]["n_points"])

    rho, eta, gamma, v_bar, v0, kappa = sample_params(cfg, n)
    surfaces = ssvi(ttms, ks, rho, eta, gamma, v_bar, v0, kappa) 

    return ttms, ks, surfaces


def data_preparation(cfg, n, n_context):
    ttms, ks, surfaces = generate_surfaces(cfg, n)

    TT, KK = np.meshgrid(ttms, ks, indexing='ij')
    k_flat   = KK.ravel()
    tau_flat = TT.ravel()

    k_idx, t_idx = sample_sparse_points(ks, ttms, n_context, n_samples=n)
    train_idx = t_idx * len(ks) + k_idx

    train, test = [], []
    for i in range(n):
        sigma = surfaces[i].ravel()

        test_idx = np.setdiff1d(np.arange(len(sigma)), train_idx[i])

        X_train, y_train = np.column_stack([k_flat[train_idx[i]], tau_flat[train_idx[i]]]), sigma[train_idx[i]]
        X_test,  y_test  = np.column_stack([k_flat[test_idx],tau_flat[test_idx]]), sigma[test_idx]

        train.append((X_train, y_train))
        test.append((X_test,  y_test))

    return train, test


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))
    train, test = data_preparation(cfg, 1)
    from tabpfn import TabPFNRegressor

    X_train, y_train = train[0]
    X_test,  y_test  = test[0]

    model = TabPFNRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)