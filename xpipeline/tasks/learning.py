import logging
import random
import numpy as np

# it looks strange, but this lets us write the fallback logic
# once in core and use the various array module names here
# letting them be `None` without it being an ImportError
# (We'll see if that was a good decision...)
from .. import core
cp = core.cupy
da = core.dask_array
torch = core.torch

from scipy.sparse.linalg import aslinearoperator, svds

import dask


from .. import utils

log = logging.getLogger(__name__)

@dask.delayed(nout=2)
def train_test_split(array, test_frac, random_state=0):
    random.seed(random_state)
    log.debug(f'{array=}')
    n_elems = array.shape[0]
    elements = range(n_elems)
    mask = np.zeros(n_elems, dtype=bool)
    test_count = int(test_frac * n_elems)
    test_elems = random.choices(elements, k=test_count)
    for idx in test_elems:
        mask[idx] = True
    train_subarr, test_subarr = array[~mask], array[mask]
    log.info(f"Cross-validation reserved {100 * test_frac:2.1f}% of inputs")
    log.info(f'Split {n_elems} into {train_subarr.shape[0]} and {test_subarr.shape[0]}')
    return train_subarr, test_subarr


def drop_idx_range_cols(arr, min_excluded_idx, max_excluded_idx):
    '''Note exclusive upper bound: [min_excluded_idx, max_excluded_idx)'''
    xp = core.get_array_module(arr)
    rows, cols = arr.shape
    n_drop = max_excluded_idx - min_excluded_idx
    out_shape = (rows, cols - n_drop)
    out = xp.empty(out_shape, dtype=arr.dtype)
    # L | |  R

    # L
    out[:,:min_excluded_idx] = arr[:,:min_excluded_idx]
    # R
    out[:,min_excluded_idx:] = arr[:,max_excluded_idx:]
    return out


def drop_idx_range_rows(arr, min_excluded_idx, max_excluded_idx):
    '''Note exclusive upper bound: [min_excluded_idx, max_excluded_idx)'''
    xp = core.get_array_module(arr)
    rows, cols = arr.shape
    n_drop = max_excluded_idx - min_excluded_idx
    out_shape = (rows - n_drop, cols)
    out = xp.empty(out_shape, dtype=arr.dtype)
    #  U
    # ===
    #  L

    # U
    out[:min_excluded_idx] = arr[:min_excluded_idx]
    # L
    out[min_excluded_idx:] = arr[max_excluded_idx:]
    return out


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
    torch_array = torch.as_tensor(array)
    # Note: use of the `out=` argument for torch.svd and preallocated
    # output tensors proved not to save any runtime, so for simplicity
    # they're not retained.
    torch_mtx_u, torch_diag_s, torch_mtx_v = torch.svd(torch_array, some=not full_matrices)
    mtx_u = xp.asarray(torch_mtx_u)
    diag_s = xp.asarray(torch_diag_s)
    mtx_v = xp.asarray(torch_mtx_v)
    if n_modes is not None:
        return mtx_u[:,:n_modes], diag_s[:n_modes], mtx_v[:,:n_modes]
    else:
        return mtx_u, diag_s, mtx_v


def generic_svd(mtx_x, n_modes, n_power_iter=4):
    '''Computes SVD of mtx_x returning U, s, and V such that
    allclose(mtx_x, U @ diag(s) @ V.T) (with some tolerance).

    When supplied with CPU arrays, `torch` is used if available, TODO
    falling back to `numpy.linalg.svd`. When supplied with GPU arrays,
    `torch` is used if available, falling back to `cupy.linalg.svd`.

    When supplied with distributed arrays, those with total number of
    elements < `dask_size_threshold` are converted to local NumPy
    arrays and processed as in the CPU array case. TODO

    Parameters
    ----------
    mtx_x : (m, n) ndarray matrix
    full_matrices : bool (default False)
        Whether to return full m x m U and full n x n V,
        otherwise U is m x r and V is n x r
        where r = min(m,n,n_modes)
    n_modes : int or None
        Whether to truncate the decomposition, keeping the top
        `n_modes` greatest singular values and corresponding vectors
    n_power_iter : int
        Number of power iterations when using
        `da.linalg.svd_compressed` to compute Halko approximate SVD.

    Returns
    -------
    mtx_u
    diag_s
    mtx_v
    '''
    xp = core.get_array_module(mtx_x)
    if xp is da:
        mtx_u, diag_s, mtx_v = da.linalg.svd_compressed(mtx_x, k=n_modes, n_power_iter=n_power_iter)
    elif xp in (np, cp):
        mtx_u, diag_s, mtx_vt = xp.linalg.svd(mtx_x, full_matrices=False)
        mtx_v = mtx_vt.T
    return mtx_u[:,:n_modes], diag_s[:n_modes], mtx_v[:,:n_modes]

def cpu_top_k_svd_arpack(array, n_modes=None):
    '''Calls scipy.sparse.linalg.svds to compute top `n_modes`
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
    '''
    if n_modes is None:
        n_modes = min(array.shape)
    mtx_u, diag_s, mtx_vt = svds(aslinearoperator(array), k=n_modes)
    return mtx_u, diag_s, mtx_vt.T

def minimal_downdate(mtx_u, diag_s, mtx_v, min_col_to_remove, max_col_to_remove, compute_v=False):
    '''Modify an existing SVD `mtx_u @ diag(diag_s) @ mtx_v.T` to
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
    '''
    xp = core.get_array_module(mtx_u)
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
    mtx_uta = -(xp.diag(diag_s) @ mtx_v[min_col_to_remove:max_col_to_remove].T)
    # and just the rows of V corresponding to removed columns:
    mtx_vtb = mtx_v[min_col_to_remove:max_col_to_remove].T

    # Additive modification to inner diagonal matrix
    mtx_k = xp.diag(diag_s)
    mtx_k += mtx_uta @ mtx_vtb.T  # U^T A is r x c, (V^T B)^T is c x r, O(r c r) -> r x r

    # Smaller (dimension r x r) SVD to re-diagonalize, giving
    # rotations of the left and right singular vectors and
    # updated singular values
    mtx_uprime, diag_sprime, mtx_vprime = generic_svd(mtx_k, n_modes=dim_r)

    # Compute new SVD by applying the rotations
    new_mtx_u = mtx_u @ mtx_uprime
    new_diag_s = diag_sprime
    if compute_v:
        new_mtx_v = mtx_v @ mtx_vprime
        # columns of X become rows of V, delete the dropped ones
        new_mtx_v = drop_idx_range_rows(new_mtx_v, min_col_to_remove, max_col_to_remove)
    else:
        new_mtx_v = None
    return new_mtx_u, new_diag_s, new_mtx_v
