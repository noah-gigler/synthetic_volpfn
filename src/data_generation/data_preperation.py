import numpy as np
from src.data_generation.SSVI import ssvi, sample_params

# gaussian z sampling with mean ATM (z=0)
# uniform ttm sampling as the tau grid is already denser at the short end (rho=sqrt(tau) uniform)
def sample_sparse_points(zs, ttms, n_points, n_samples):
    z_weights = np.exp(-0.5 * (zs / 0.25) ** 2)           # ATM-weighted in z (z=0 is ATM)
    z_weights /= z_weights.sum()

    flat_weights = np.tile(z_weights, len(ttms))
    flat_weights /= flat_weights.sum()

    scalar = np.isscalar(n_points)
    n_points = np.broadcast_to(np.asarray(n_points, dtype=int), (n_samples,))

    flat_idx = [
        np.random.choice(len(zs) * len(ttms), size=m, replace=False, p=flat_weights)
        for m in n_points
    ]

    if scalar:
        t_idx, k_idx = np.unravel_index(np.array(flat_idx), (len(ttms), len(zs)))
        return k_idx, t_idx

    pairs = [np.unravel_index(fi, (len(ttms), len(zs))) for fi in flat_idx]
    return [p[1] for p in pairs], [p[0] for p in pairs]


def sample_context_sizes(n_context, n, dist="uniform", group=1):
    # group>1: one size per group of surfaces (marginal dist unchanged) so equal-size
    # groups can share a single batched forward pass in finetuning
    if np.isscalar(n_context):
        return np.full(n, n_context, dtype=int)
    lo, hi = n_context
    m = -(-n // group)
    if dist == "uniform":
        sizes = np.random.randint(lo, hi + 1, size=m)
    else:
        u = np.random.uniform(np.log(lo), np.log(hi + 1), size=m)
        sizes = np.minimum(np.exp(u).astype(int), hi)
    return np.repeat(sizes, group)[:n]


def grid_from_cfg(cfg):
    # (ρ, z) grid: ρ=√τ uniform (τ quadratic-spaced), z uniform from the config k-block
    rho = np.linspace(np.sqrt(cfg["ttm"]["min"]), np.sqrt(cfg["ttm"]["max"]), cfg["ttm"]["n_points"])
    zs = np.linspace(cfg["z"]["min"], cfg["z"]["max"], cfg["z"]["n_points"])
    return rho**2, zs


def grid_points(cfg):
    # flattened physical grid (ttm-major); standardized moneyness z mapped to k = z·√τ
    ttms, zs = grid_from_cfg(cfg)
    TT, ZZ = np.meshgrid(ttms, zs, indexing="ij")
    return (ZZ * np.sqrt(TT)).ravel(), TT.ravel()


def generate_surfaces(cfg, n):
    ttms, zs = grid_from_cfg(cfg)
    KK = zs[None, :] * np.sqrt(ttms[:, None])          # (n_ttm, n_z) wedge, k = z·√τ

    rho, eta, gamma, v_bar, v0, kappa = sample_params(cfg, n)
    surfaces = ssvi(ttms, KK, rho, eta, gamma, v_bar, v0, kappa)

    return ttms, zs, surfaces


def _split_context_query(zs, ttms, surfaces, k_idx, t_idx):
    TT, ZZ = np.meshgrid(ttms, zs, indexing='ij')
    KK = ZZ * np.sqrt(TT)                              # k = z·√τ
    k_flat   = KK.ravel()
    tau_flat = TT.ravel()

    train, test = [], []
    for i in range(len(surfaces)):
        sigma = surfaces[i].ravel()
        train_idx = t_idx[i] * len(zs) + k_idx[i]

        # query is the full grid, including context points
        X_train, y_train = np.column_stack([k_flat[train_idx], tau_flat[train_idx]]), sigma[train_idx]
        X_test,  y_test  = np.column_stack([k_flat, tau_flat]), sigma

        train.append((X_train, y_train))
        test.append((X_test,  y_test))

    return train, test


def data_preparation(cfg, n, n_context, size_dist="uniform", size_group=1):
    ttms, zs, surfaces = generate_surfaces(cfg, n)
    sizes = sample_context_sizes(n_context, n, dist=size_dist, group=size_group)
    k_idx, t_idx = sample_sparse_points(zs, ttms, sizes, n_samples=n)
    return _split_context_query(zs, ttms, surfaces, k_idx, t_idx)


def make_stratified_eval_set(cfg, n_surfaces, context_sizes):
    ttms, zs, surfaces = generate_surfaces(cfg, n_surfaces)

    train, test = [], []
    for size in context_sizes:
        k_idx, t_idx = sample_sparse_points(zs, ttms, np.full(n_surfaces, size), n_samples=n_surfaces)
        tr, te = _split_context_query(zs, ttms, surfaces, k_idx, t_idx)
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