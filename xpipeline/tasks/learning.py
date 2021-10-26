import logging
import random
import numpy as np
from .. import core
cp = core.cupy
from scipy import linalg

from scipy.sparse.linalg import aslinearoperator, svds

import numba


from .. import utils

log = logging.getLogger(__name__)


def train_test_split(array, test_frac, random_state=0):
    random.seed(random_state)
    log.debug(f"{array=}")
    n_elems = array.shape[0]
    elements = range(n_elems)
    mask = np.zeros(n_elems, dtype=bool)
    test_count = int(test_frac * n_elems)
    test_elems = random.choices(elements, k=test_count)
    for idx in test_elems:
        mask[idx] = True
    train_subarr, test_subarr = array[~mask], array[mask]
    log.info(f"Cross-validation reserved {100 * test_frac:2.1f}% of inputs")
    log.info(f"Split {n_elems} into {train_subarr.shape[0]} and {test_subarr.shape[0]}")
    return train_subarr, test_subarr

def torch_svd(array, full_matrices=False, n_modes=None):
    '''Wrap `torch.svd` to handle conversion between NumPy/CuPy arrays
    and Torch tensors. Returns U s V such that
    allclose(U @ diag(s) @ V.T, array) (with some tolerance).

    Parameters
    ----------
    array : (m, n) array
    full_matrices : bool (default False)
        Whether to return full m x m U and full n x n V,
        otherwise U is m x r and V is n x r
        where r = min(m,n,n_modes)
    n_modes : int or None
        Whether to truncate the decomposition, keeping the top
        `n_modes` greatest singular values and corresponding vectors

    Returns
    -------
    mtx_u
    diag_s
    mtx_v
    '''
    xp = core.get_array_module(array)
    torch_array = core.torch.as_tensor(array)
    # Note: use of the `out=` argument for torch.svd and preallocated
    # output tensors proved not to save any runtime, so for simplicity
    # they're not retained.
    torch_mtx_u, torch_diag_s, torch_mtx_v = core.torch.svd(torch_array, some=not full_matrices)
    mtx_u = xp.asarray(torch_mtx_u)
    diag_s = xp.asarray(torch_diag_s)
    mtx_v = xp.asarray(torch_mtx_v)
    if n_modes is not None:
        return mtx_u[:,:n_modes], diag_s[:n_modes], mtx_v[:,:n_modes]
    else:
        return mtx_u, diag_s, mtx_v

def generic_svd(mtx_x, n_modes=None, full_matrices=False):
    """Computes SVD of mtx_x returning U, s, and V such that
    allclose(mtx_x, U @ diag(s) @ V.T) (with some tolerance).

    If `n_modes` is not None, this function will truncate the
    return values to the first `n_modes` columns/entries, and
    copy if necessary to make the array contiguous.

    When supplied with CPU arrays, `numpy.linalg.svd` is called.
    When supplied with GPU arrays, `cupy.linalg.svd` is used.

    Parameters
    ----------
    mtx_x : (m, n) ndarray matrix
    n_modes : int or None (default None)
        Whether to truncate the decomposition, keeping the top
        `n_modes` greatest singular values and corresponding vectors
    full_matrices : bool (default False)
        Whether to return full m x m U and full n x n V,
        otherwise U is m x r and V is n x r
        where r = min(m,n,n_modes)
    Returns
    -------
    mtx_u
    diag_s
    mtx_v
    """
    xp = core.get_array_module(mtx_x)
    mtx_u, diag_s, mtx_vt = xp.linalg.svd(mtx_x, full_matrices=full_matrices)
    mtx_v = mtx_vt.T
    return (
        xp.ascontiguousarray(mtx_u[:, :n_modes]),
        xp.ascontiguousarray(diag_s[:n_modes]),
        xp.ascontiguousarray(mtx_v[:, :n_modes])
    )

def eigh_top_k(mtx, k_klip):
    n = mtx.shape[0]
    lambda_values_out, mtx_c_out = linalg.eigh(mtx, subset_by_index=[n - k_klip, n - 1])
    lambda_values = np.flip(lambda_values_out)[:k_klip]
    mtx_c = np.flip(mtx_c_out, axis=1)[:,:k_klip]
    return lambda_values, mtx_c

def eigh_full_decomposition(mtx, k_klip):
    lambda_values_out, mtx_c_out = linalg.eigh(mtx, np.eye(mtx.shape[0]), driver="gvd")
    # flip so evals are descending, truncate to k_klip
    lambda_values = np.flip(lambda_values_out)[:k_klip]
    mtx_c = np.flip(mtx_c_out, axis=1)[:,:k_klip]
    return lambda_values, mtx_c

def cpu_top_k_svd_arpack(array, n_modes=None):
    """Calls scipy.sparse.linalg.svds to compute top `n_modes`
    singular vectors.  Returns U s V such that
    `U @ diag(s) @ V.T` is the rank-`n_modes` SVD of `array`

    Parameters
    ----------
    array : (m, n) array
    n_modes : int or None
        Compute `n_modes` greatest singular values
        and corresponding vectors, or else default to
        ``n_modes = min(m,n)``

    Returns
    -------
    mtx_u
    diag_s
    mtx_v
    """
    if n_modes is None:
        n_modes = min(array.shape)
    mtx_u, diag_s, mtx_vt = svds(array, k=n_modes)
    sorter = np.flip(np.argsort(diag_s))
    mtx_u, diag_s, mtx_v = (
        np.ascontiguousarray(mtx_u[:, sorter]),
        diag_s[sorter],
        np.ascontiguousarray(mtx_vt.T[:, sorter])
    )
    return mtx_u, diag_s, mtx_v


@numba.njit(inline='always')
def _numba_svd_wrap(mtx_x, n_modes):
    mtx_u, diag_s, mtx_vt = np.linalg.svd(mtx_x, full_matrices=False)
    mtx_v = mtx_vt.T
    return np.ascontiguousarray(mtx_u[:, :n_modes]), np.ascontiguousarray(diag_s[:n_modes]), np.ascontiguousarray(mtx_v[:, :n_modes])

@numba.njit(cache=True)
def minimal_downdate(
    mtx_u, diag_s, mtx_v, min_col_to_remove, max_col_to_remove, compute_v=False
):
    """Modify an existing SVD `mtx_u @ diag(diag_s) @ mtx_v.T` to
    remove columns given by `col_idxs_to_remove`, returning
    a new diagonalization

    Attempts to use the "right" SVD implementation for the input types,
    see `generic_svd` docs for details

    Parameters
    ----------
    mtx_u : array (p, r)
    diag_s : array (r,)
    mtx_v : array (q, r)
    min,max_col_to_remove : [min, max) interval to drop
    compute_v : bool (default False)
        Multiply out the new right singular vectors instead
        of discarding their rotation

    Returns
    -------
    new_mtx_u : array (p, r)
    new_diag_s : array (r,)
    new_mtx_v : array (q, r) or None
        If `compute_v` is True this is the updated V matrix,
        otherwise None.
    """
    dim_r = diag_s.shape[0]
    assert mtx_u.shape[1] == dim_r
    assert mtx_v.shape[1] == dim_r

    # Omit computation of mtx_a and mtx_b as their products
    # mtx_uta and mtx_vtb can be expressed without the intermediate
    # arrays.

    # Omit computation of P, R_A, Q, R_B
    # as they represent the portion of the update matrix AB^T
    # not captured in the original basis and we're making
    # the assumption that downdating our (potentially truncated)
    # SVD doesn't require new basis vectors, merely rotating the
    # existing ones. Indeed, P, R_A, Q, and R_B are very close to
    # machine zero

    # "Eigen-code" the update matrices from both sides
    # into the space where X is diagonalized (and truncated)
    #
    # This is just the first part of the product that would have been
    # formed to make mtx_a:
    mtx_v_subset = np.ascontiguousarray(mtx_v[min_col_to_remove:max_col_to_remove].T) # make contiguous so Numba's happy
    mtx_uta = -(np.diag(diag_s) @ mtx_v_subset)
    # and just the rows of V corresponding to removed columns:
    mtx_vtb = mtx_v_subset

    # Additive modification to inner diagonal matrix
    mtx_k = np.diag(diag_s)
    mtx_k += (
        mtx_uta @ mtx_vtb.T
    )  # U^T A is r x c, (V^T B)^T is c x r, O(r c r) -> r x r

    # Smaller (dimension r x r) SVD to re-diagonalize, giving
    # rotations of the left and right singular vectors and
    # updated singular values
    mtx_uprime, diag_sprime, mtx_vprime = _numba_svd_wrap(mtx_k, n_modes=dim_r)

    # Compute new SVD by applying the rotations
    new_mtx_u = mtx_u @ mtx_uprime  # make contiguous so Numba's happy
    new_diag_s = diag_sprime
    if compute_v:
        new_mtx_v = mtx_v @ mtx_vprime
        # columns of X become rows of V, delete the dropped ones
        new_mtx_v = utils.drop_idx_range_rows(new_mtx_v, min_col_to_remove, max_col_to_remove)
    else:
        new_mtx_v = None
    return new_mtx_u, new_diag_s, new_mtx_v
