import numpy as np
from src.data_generation.SSVI import ssvi, sample_params
from src.data_generation.heston import heston, sample_params as heston_sample_params
from src.data_generation.grid import Grid

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


def generate_surfaces(cfg, n, heston_frac=None):
    # heston_frac: fraction of surfaces drawn from Heston instead of SSVI - mixing families
    # encourages a genuine prior over arbitrage-free shapes rather than overfitting to one
    # parametric form (see VolSmoothing_with_TabPFN_proposal.pdf). Defaults to cfg's
    # mixture.heston_frac if set, else 0.0 (pure SSVI, unchanged from before this existed) -
    # every existing caller (noise.py's clean/noisy/eval-set builders) goes through this
    # function with no extra args, so the default must not silently change their behavior.
    g = Grid(cfg)
    if heston_frac is None:
        heston_frac = cfg.get("mixture", {}).get("heston_frac", 0.0)
    n_heston = int(round(n * heston_frac))
    n_ssvi = n - n_heston

    surfaces = np.empty((n,) + g.shape)
    if n_ssvi:
        rho, eta, gamma, v_bar, v0, kappa = sample_params(cfg, n_ssvi)
        surfaces[:n_ssvi] = ssvi(g.ttms, g.k.reshape(g.shape), rho, eta, gamma, v_bar, v0, kappa)
    if n_heston:
        v0h, kappah, thetah, sigmah, rhoh = heston_sample_params(cfg, n_heston)
        # a small fraction of grid points (typically <1%, deep-OTM/shortest-maturity corner)
        # come back NaN - a genuine float64 price-underflow floor, not a bug (report_notes.md).
        # Left as NaN, not imputed: TabPFN's own finetuning loss (_compute_regression_loss's
        # CRPS/MSE terms, this project's default) already masks NaN targets to exactly zero
        # loss contribution rather than propagating them - confirmed by reading its source.
        # Filling with a fake value (e.g. the surface's own mean IV) would be actively wrong:
        # it injects a flat, physically incorrect value into exactly the most extreme corner of
        # the surface, corrupting the training signal there instead of correctly excluding it.
        surfaces[n_ssvi:] = heston(g.ttms, g.k.reshape(g.shape), v0h, kappah, thetah, sigmah, rhoh)

    order = np.random.permutation(n)  # SSVI/Heston blocks shuffled together, not left contiguous
    return g, surfaces[order]


def _split_context_query(g, surfaces, k_idx, t_idx):
    # model feature is (z, tau); the wedge structure lives in z, so feed z not k = z·√τ
    train, test = [], []
    for i in range(len(surfaces)):
        sigma = surfaces[i].ravel()
        train_idx = t_idx[i] * g.shape[1] + k_idx[i]

        # query is the full grid, including context points
        X_train, y_train = np.column_stack([g.z[train_idx], g.tau[train_idx]]), sigma[train_idx]
        X_test,  y_test  = g.features(), sigma

        train.append((X_train, y_train))
        test.append((X_test,  y_test))

    return train, test


def data_preparation(cfg, n, n_context, size_dist="uniform", size_group=1):
    g, surfaces = generate_surfaces(cfg, n)
    sizes = sample_context_sizes(n_context, n, dist=size_dist, group=size_group)
    k_idx, t_idx = sample_sparse_points(g.zs, g.ttms, sizes, n_samples=n)
    return _split_context_query(g, surfaces, k_idx, t_idx)


def make_stratified_eval_set(cfg, n_surfaces, context_sizes):
    g, surfaces = generate_surfaces(cfg, n_surfaces)

    train, test = [], []
    for size in context_sizes:
        k_idx, t_idx = sample_sparse_points(g.zs, g.ttms, np.full(n_surfaces, size), n_samples=n_surfaces)
        tr, te = _split_context_query(g, surfaces, k_idx, t_idx)
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