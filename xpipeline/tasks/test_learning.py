import numpy as np
from .learning import train_test_split, minimal_downdate
from ..utils import drop_idx_range_cols


def test_train_test_split():
    n_total = 10
    data = np.arange(n_total)
    train_a, test_a = train_test_split(data, 0.5, random_state=0)
    assert train_a.size + test_a.size == data.size, "Lost elements in split"
    assert train_a.shape[0] == n_total // 2, "Wrong split"
    assert test_a.shape[0] == n_total // 2, "Wrong split"
    repeat_train_a, repeat_test_a = train_test_split(data, 0.5, random_state=0)
    assert np.all(
        train_a == repeat_train_a
    ), "Repeat call with same RNG state not reproducible"
    assert np.all(
        test_a == repeat_test_a
    ), "Repeat call with same RNG state not reproducible"
    train_b, test_b = train_test_split(data, 0.5, random_state=1)
    assert np.any(train_a != train_b), "Repeat call with new RNG state not different"
    assert np.any(test_a != test_b), "Repeat call with new RNG state not different"


def compare_columns_modulo_sign(init_u, final_u, display=False):
    signs = np.zeros(init_u.shape[1])
    for col in range(init_u.shape[1]):
        signs[col] = 1 if np.allclose(init_u[:, col], final_u[:, col]) else -1
    vmax = np.max(np.abs([init_u, final_u]))
    final_u_mod = signs * final_u
    if display:
        import matplotlib.pyplot as plt

        fig, (ax_iu, ax_fu, ax_du) = plt.subplots(ncols=3, figsize=(14, 4))
        ax_iu.imshow(init_u, vmin=-vmax, vmax=vmax, origin="lower")
        ax_iu.set_title(r"$\mathbf{U}_\mathrm{first}$")
        ax_fu.imshow(final_u_mod, vmin=-vmax, vmax=vmax, origin="lower")
        ax_fu.set_title(r"(signs) * $\mathbf{U}_\mathrm{second}$")
        diff_vmax = np.max(np.abs(final_u_mod - init_u))
        ax_du.imshow(
            final_u_mod - init_u,
            cmap="RdBu_r",
            vmax=diff_vmax,
            vmin=-diff_vmax,
            origin="lower",
        )
        ax_du.set_title(
            r"(signs) * $\mathbf{U}_\mathrm{second}$ - $\mathbf{U}_\mathrm{first}$"
        )
    return np.allclose(final_u_mod, init_u)


def test_minimal_downdate(epsilon=1e-14):
    dim_p = 6
    dim_q = 5

    # Initialize p x q noise matrix X
    mtx_x = np.random.randn(dim_p, dim_q)

    # Initialize thin SVD
    mtx_u, diag_s, mtx_vt = np.linalg.svd(mtx_x, full_matrices=False)
    mtx_v = mtx_vt.T

    # Select columns to remove
    min_col_to_remove = 1
    max_col_to_remove = 3  # exclusive
    new_mtx_u, new_diag_s, new_mtx_v = minimal_downdate(
        mtx_u, diag_s, mtx_v, min_col_to_remove, max_col_to_remove, compute_v=True
    )

    # X with columns zeroed for comparison
    final_mtx_x = drop_idx_range_cols(mtx_x, min_col_to_remove, max_col_to_remove)
    assert np.allclose(
        new_mtx_u @ np.diag(new_diag_s) @ new_mtx_v.T, final_mtx_x, atol=1e-6
    )

    # SVD of final matrix for comparison
    final_mtx_u, final_diag_s, final_mtx_vt = np.linalg.svd(final_mtx_x)

    n_nonzero = np.count_nonzero(final_diag_s > epsilon)
    assert n_nonzero == 3

    assert compare_columns_modulo_sign(
        new_mtx_u[:, :n_nonzero],
        final_mtx_u[:, :n_nonzero],
    )
