from enum import Enum
import dask
from dataclasses import dataclass
from typing import Union, Optional
from collections.abc import Callable
import numpy as np
import dask.array as da
import distributed.protocol
from scipy import linalg
from numba import njit
from .. import core, utils, constants
from . import learning, improc

@njit
def _count_max_excluded(values, delta_excluded):
    max_excluded = 0
    n_values = values.shape[0]
    for i in range(n_values):
        excluded_for_val1 = 0
        val1 = values[i]
        for j in range(n_values):
            delta = values[j] - val1
            if -delta_excluded <= delta <= delta_excluded:
                excluded_for_val1 += 1
        max_excluded = excluded_for_val1 if excluded_for_val1 > max_excluded else max_excluded
    return max_excluded


@dataclass
class KlipInput:
    sci_arr: da.core.Array
    estimation_mask: np.ndarray
    combination_mask: Union[np.ndarray, None]
distributed.protocol.register_generic(KlipInput)

@dataclass
class ExclusionValues:
    exclude_within_delta : Union[int,float]
    values : np.ndarray
    num_excluded_max : Optional[int] = None  # Downdate depends on knowing ahead of time how many frames you'll need to remove

    def __post_init__(self):
        if self.num_excluded_max is None:
            self.num_excluded_max = _count_max_excluded(self.values, self.exclude_within_delta)
            log.debug(f'initialized {self.num_excluded_max=} for {self.values=}')

    def get_mask(self, current_value=None):
        deltas = np.abs(self.values - current_value)
        return deltas <= self.exclude_within_delta

distributed.protocol.register_generic(ExclusionValues)

@dataclass
class KlipParams:
    k_klip: int
    exclusions : list[ExclusionValues]
    decomposer : Callable
    reuse : bool = False
    initial_decomposer : Optional[Callable] = None
    missing_data_value: float = np.nan
    strategy : constants.KlipStrategy = constants.KlipStrategy.DOWNDATE_SVD
    chunks : Optional[Union[tuple, int, dict]] = None

    def __post_init__(self):
        if self.initial_decomposer is None:
            self.initial_decomposer = self.decomposer
distributed.protocol.register_generic(KlipParams)

import logging

log = logging.getLogger(__name__)


def drop_idx_range_cols(arr, min_excluded_idx, max_excluded_idx):
    '''Note exclusive upper bound: [min_excluded_idx, max_excluded_idx)'''
    return drop_idx_range_rows(arr.T, min_excluded_idx, max_excluded_idx).T

def drop_idx_range_rows(arr, min_excluded_idx, max_excluded_idx):
    '''Note exclusive upper bound: [min_excluded_idx, max_excluded_idx)'''
    xp = core.get_array_module(arr)
    rows, cols = arr.shape
    if min_excluded_idx < 0:
        raise ValueError("Negative indexing unsupported")
    if max_excluded_idx > rows or max_excluded_idx < min_excluded_idx:
        raise ValueError("Upper limit of excluded indices out of bounds")
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

def mean_subtract_vecs(image_vecs):
    xp = core.get_array_module(image_vecs)
    mean_vec = xp.average(image_vecs, axis=1)
    image_vecs_meansub = image_vecs - mean_vec[:, core.newaxis]
    return image_vecs_meansub, mean_vec


class Decomposer:
    def __init__(self, image_vecs, n_modes):
        self.image_vecs = image_vecs
        self.meansub_image_vecs, self.mean_vec = mean_subtract_vecs(image_vecs)
        self.n_modes = n_modes
        self.xp = core.get_array_module(image_vecs)
        self.idxs = self.xp.arange(self.meansub_image_vecs.shape[1])

    def eigenimages(self, min_excluded_idx, max_excluded_idx):
        raise NotImplementedError()


class SVDDecomposer(Decomposer):
    def eigenimages(self, min_excluded_idx, max_excluded_idx):
        ref = drop_idx_range_cols(
            self.meansub_image_vecs, min_excluded_idx, max_excluded_idx
        )
        u, s, v = learning.generic_svd(ref, n_modes=self.n_modes)
        return u[:, : self.n_modes]


class MinimalDowndateSVDDecomposer(Decomposer):
    def __init__(self, image_vecs, n_modes, extra_modes=1):
        super().__init__(image_vecs, n_modes)
        self.n_modes = n_modes
        self.mtx_u, self.diag_s, self.mtx_v = learning.generic_svd(
            self.meansub_image_vecs, n_modes=n_modes + extra_modes
        )
        self.idxs = self.xp.arange(image_vecs.shape[1])

    def eigenimages(self, min_excluded_idx, max_excluded_idx):
        new_u, new_s, new_v = learning.minimal_downdate(
            self.mtx_u,
            self.diag_s,
            self.mtx_v,
            min_col_to_remove=min_excluded_idx,
            max_col_to_remove=max_excluded_idx,
        )
        return new_u[:, : self.n_modes]


def klip_frame(target, decomposer, exclude_idx_min, exclude_idx_max):
    eigenimages = decomposer.eigenimages(exclude_idx_min, exclude_idx_max)
    meansub_target = target - decomposer.mean_vec
    return meansub_target - eigenimages @ (eigenimages.T @ meansub_target)

def klip_to_modes(image_vecs, decomp_class, n_modes, exclude_nearest=0):
    """
    Parameters
    ----------
    image_vecs : array (m, n)
        a series of n images arranged into columns of m pixels each
        and mean subtracted such that each pixel-series mean is 0
    decomp_class : Decomposer subclass
        Must accept image_vecs, n_modes, solver in __init__ and implement
        eigenimages() method
    n_modes : int
        Rank of low-rank decomposition
    exclude_nearest : int
        In addition to excluding the current frame, exclude
        this many adjacent frames. (Note the first and last
        few frames won't exclude the same number of frames;
        they will just go to the ends of the dataset.)
    """
    xp = core.get_array_module(image_vecs)
    _, n_frames = image_vecs.shape

    output = xp.zeros_like(image_vecs)
    idxs = xp.arange(image_vecs.shape[1])
    decomposer = decomp_class(image_vecs, n_modes)
    for i in range(image_vecs.shape[1]):
        output[:, i] = klip_frame(
            image_vecs[:, i],
            decomposer,
            exclude_idx_min=max(i - exclude_nearest, 0),
            exclude_idx_max=min(i + 1 + exclude_nearest, n_frames),
        )
    return output


def klip_mtx(image_vecs, params : KlipParams):
    xp = core.get_array_module(image_vecs)
    image_vecs_meansub, _ = mean_subtract_vecs(image_vecs)
    if params.strategy in (constants.KlipStrategy.DOWNDATE_SVD, constants.KlipStrategy.SVD):
        if xp is da:
            if params.chunks is None:
                # For the Dask chunked exact SVD:
                # "The memory constraints here are that if you have an n by m tall
                # and skinny array (n >> m) cut into k blocks then you need to have
                # about m**2 * k space."
                # -- https://blog.dask.org/2020/05/13/large-svds
                if image_vecs_meansub.shape[0] > image_vecs_meansub.shape[1]:
                    chunk_request = {0: "auto", 1: -1}
                else:
                    chunk_request = {0: -1, 1: "auto"}
            else:
                chunk_request = params.chunks
            image_vecs_meansub = image_vecs_meansub.rechunk(chunk_request)
            log.debug(f"Rechunked {image_vecs_meansub.shape=} into {image_vecs_meansub.numblocks=}")
        return klip_mtx_svd(image_vecs_meansub, params)
    elif params.strategy is constants.KlipStrategy.COVARIANCE:
        return klip_mtx_covariance(image_vecs_meansub, params)
    else:
        raise ValueError(f"Unknown strategy value in {params=}")

def drop_idx_range_rows_cols(arr, min_excluded_idx, max_excluded_idx):
    '''Note exclusive upper bound: [min_excluded_idx, max_excluded_idx)'''
    xp = core.get_array_module(arr)
    rows, cols = arr.shape
    assert rows == cols
    n_drop = max_excluded_idx - min_excluded_idx
    out_shape = (rows - n_drop, cols - n_drop)
    out = xp.empty(out_shape, dtype=arr.dtype)
    # UL | | U R
    # ===   ====
    # LL | | L R

    # UL
    out[:min_excluded_idx,:min_excluded_idx] = arr[:min_excluded_idx,:min_excluded_idx]
    # UR
    out[:min_excluded_idx,min_excluded_idx:] = arr[:min_excluded_idx,max_excluded_idx:]
    # LL
    out[min_excluded_idx:,:min_excluded_idx] = arr[max_excluded_idx:,:min_excluded_idx]
    # LR
    out[min_excluded_idx:,min_excluded_idx:] = arr[max_excluded_idx:,max_excluded_idx:]
    return out


def _eigh_full_decomposition(mtx, k_klip):
    lambda_values_out, mtx_c_out = linalg.eigh(mtx, np.eye(mtx.shape[0]), driver="gvd")
    # flip so evals are descending, truncate to k_klip
    lambda_values = np.flip(lambda_values_out)[:k_klip]
    mtx_c = np.flip(mtx_c_out, axis=1)[:,:k_klip]
    return lambda_values, mtx_c

def _eigh_top_k(mtx, k_klip):
    n = mtx.shape[0]
    lambda_values_out, mtx_c_out = linalg.eigh(mtx, subset_by_index=[n - k_klip, n - 1])
    lambda_values = np.flip(lambda_values_out)[:k_klip]
    mtx_c = np.flip(mtx_c_out, axis=1)[:,:k_klip]
    return lambda_values, mtx_c

def _klip_mtx_covariance(image_vecs_meansub : np.ndarray, params : KlipParams):
    '''Apply KLIP to mean-subtracted image column vectors

    Parameters
    ----------
    image_vecs_meansub : np.ndarray
        image vectors, one column per image
    params : KlipParams
        configuration for tunable parameters
    '''
    k_klip = params.k_klip

    output = np.zeros_like(image_vecs_meansub)
    mtx_e_all = image_vecs_meansub.T @ image_vecs_meansub
    n_images = image_vecs_meansub.shape[1]
    if params.reuse:
        lambda_values, mtx_c = params.decomposer(mtx_e_all, k_klip)
        eigenimages = image_vecs_meansub @ (mtx_c * np.power(lambda_values, -1/2))
    for i in range(n_images):
        if not params.reuse:
            min_excluded_idx, max_excluded_idx = exclusions_to_range(
                n_images=n_images,
                current_idx=i,
                exclusions=params.exclusions,
            )
            min_excluded_idx = max(min_excluded_idx, 0)
            max_excluded_idx = min(n_images, max_excluded_idx)
            mtx_e = drop_idx_range_rows_cols(mtx_e_all, min_excluded_idx, max_excluded_idx)
            lambda_values, mtx_c = params.decomposer(mtx_e, k_klip)
            ref_subset = np.hstack((image_vecs_meansub[:,:min_excluded_idx], image_vecs_meansub[:,max_excluded_idx:]))
            eigenimages = ref_subset @ (mtx_c * np.power(lambda_values, -1/2))

        meansub_target = image_vecs_meansub[:,i]
        output[:,i] = meansub_target - eigenimages @ (eigenimages.T @ meansub_target)
    return output

def klip_mtx_covariance(image_vecs_meansub, params):
    xp = core.get_array_module(image_vecs_meansub)
    if xp is da:
        d_output = dask.delayed(_klip_mtx_covariance)(image_vecs_meansub, params)
        output = da.from_delayed(
            d_output,
            shape=image_vecs_meansub.shape,
            dtype=image_vecs_meansub.dtype
        )
    else:
        output = _klip_mtx_covariance(image_vecs_meansub, params)
    return output

def exclusions_to_mask(n_images, current_idx, exclusions):
    mask = np.zeros(n_images, dtype=bool)
    mask[current_idx] = True  # exclude at least the current frame
    # mask[i] = True
    for excl in exclusions:
        individual_mask = excl.get_mask(excl.values[current_idx])
        # log.debug(f'Masking {np.count_nonzero(individual_mask)} from {excl}')
        mask |= individual_mask
    return mask

def exclusions_to_range(n_images, current_idx, exclusions):
    mask = exclusions_to_mask(n_images, current_idx, exclusions)
    indices = np.argwhere(mask)
    min_excluded_idx = np.min(indices)
    max_excluded_idx = np.max(indices)
    # detect if this isn't a simple range (which we can't handle)
    frame_idxs = np.arange(n_images)
    contiguous_mask = (frame_idxs >= min_excluded_idx) & (frame_idxs <= max_excluded_idx)
    if np.count_nonzero(mask ^ contiguous_mask):
        raise ValueError("Non-contiguous ranges to exclude detected, but we don't handle that")
    return min_excluded_idx, max_excluded_idx

def klip_chunk_svd(
    image_vecs_meansub, n_images, chunk_image_indices, params : KlipParams, mtx_u0, diag_s0, mtx_v0
):
    k_klip = params.k_klip
    if image_vecs_meansub.shape == (0, 0):
        # called by dask to infer dtype
        return np.zeros_like(image_vecs_meansub)
    log.debug(f"Klipping a chunk {image_vecs_meansub.shape=}")
    n_frames = image_vecs_meansub.shape[1]
    frame_idxs = np.arange(n_frames)  # for detecting non-contiguous masked ranges
    output = np.zeros_like(image_vecs_meansub)
    for i in range(n_frames):
        if not params.reuse:
            min_excluded_idx, max_excluded_idx = exclusions_to_range(
                n_images=n_images,
                current_idx=i,
                exclusions=params.exclusions,
            )
            log.debug(f'Excluding from {min_excluded_idx} to {max_excluded_idx} + 1')
            if params.strategy is constants.KlipStrategy.DOWNDATE_SVD:
                new_u, _, _ = learning.minimal_downdate(
                    mtx_u0,
                    diag_s0,
                    mtx_v0,
                    min_col_to_remove=min_excluded_idx,
                    max_col_to_remove=max_excluded_idx + 1,
                )
                eigenimages = new_u[:, :k_klip]
            else:
                subset_image_vecs = drop_idx_range_cols(image_vecs_meansub, min_excluded_idx, max_excluded_idx)
                eigenimages, _, _ = learning.generic_svd(subset_image_vecs, k_klip) # TODO use driver
        else:
            eigenimages = mtx_u0
        meansub_target = image_vecs_meansub[:, i]
        output[:, i] = meansub_target - eigenimages @ (eigenimages.T @ meansub_target)
    mem_mb = utils.get_memory_use_mb()
    log.debug(f"klipped! {mem_mb} MB RAM in use")
    return output

def klip_mtx_svd(image_vecs_meansub, params : KlipParams):
    k_klip = params.k_klip
    xp = core.get_array_module(image_vecs_meansub)
    output = xp.zeros_like(image_vecs_meansub)
    total_n_frames = image_vecs_meansub.shape[1]
    indices = np.arange(total_n_frames)
    if not params.reuse and params.strategy is constants.KlipStrategy.DOWNDATE_SVD:
        extra_modes = max([1,] + [excl.num_excluded_max for excl in params.exclusions])
        # extra modes:
        # to remove 1 image vector by downdating, the initial decomposition
        # should retain k_klip + 1 singular value triplets.
        initial_k = k_klip + extra_modes
        log.debug(f'{initial_k=} from {k_klip=} and {extra_modes=}')
    else:
        initial_k = k_klip
    log.debug(
        f"{image_vecs_meansub.shape=}, {k_klip=}, {initial_k=}"
    )
    if image_vecs_meansub.shape[0] < initial_k or image_vecs_meansub.shape[1] < initial_k:
        raise ValueError(f"Number of modes requested exceeds dimensions of input")
    mtx_u0, diag_s0, mtx_v0 = learning.generic_svd(image_vecs_meansub, initial_k)
    log.debug(
        f"{image_vecs_meansub.shape=}, {mtx_u0.shape=}, {diag_s0.shape=}, {mtx_v0.shape=}"
    )
    if xp is da:
        log.debug('Taking Dask path')
        if image_vecs_meansub.numblocks[0] != 1:
            image_vecs_meansub = image_vecs_meansub.rechunk({0: -1, 1: "auto"})
        output = da.blockwise(
            klip_chunk_svd,
            "ij",
            image_vecs_meansub,
            "ij",
            total_n_frames,
            None,
            indices,
            "j",
            params,
            None,
            mtx_u0,
            None,
            diag_s0,
            None,
            mtx_v0,
            None,
        )
    else:
        log.debug('Taking pure NumPy path')
        output = klip_chunk_svd(
            image_vecs_meansub,
            total_n_frames,
            indices,
            params,
            mtx_u0,
            diag_s0,
            mtx_v0,
        )
    return output


def make_good_pix_mask(
    shape, inner_radius=None, outer_radius=None, center=None, existing_mask=None
):
    if existing_mask is not None:
        if existing_mask.shape != shape:
            raise ValueError(f"Cannot mix {shape=} and {existing_mask.shape=}")
        mask = existing_mask.copy()
    else:
        mask = np.ones(shape, dtype=bool)

    if center is None:
        center = (shape[1] - 1) / 2, (shape[0] - 1) / 2

    if any(map(lambda x: x is None, (outer_radius, inner_radius))):
        rho, phi = improc.polar_coords(center, shape)
        if inner_radius is not None:
            mask &= rho >= inner_radius
        if outer_radius is not None:
            mask &= rho <= outer_radius

    return mask

DEFAULT_DECOMPOSERS = {
    constants.KlipStrategy.DOWNDATE_SVD: learning.generic_svd,
    constants.KlipStrategy.SVD: learning.generic_svd,
    constants.KlipStrategy.COVARIANCE: _eigh_top_k,
}
