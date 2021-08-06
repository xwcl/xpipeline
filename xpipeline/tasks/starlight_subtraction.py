from enum import Enum
import dask
from dataclasses import dataclass
from typing import Union, Optional
from collections.abc import Callable
import numpy as np
import dask.array as da
import distributed.protocol
from numba import njit
import numba
import time
from .. import core, utils, constants
from . import learning, improc

@njit(cache=True)
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


distributed.protocol.register_generic(ExclusionValues)


@dataclass
class InitialDecomposition:
    mtx_u0 : np.ndarray
    diag_s0 : np.ndarray
    mtx_v0 : np.ndarray
    image_vecs_meansub : np.ndarray

@dataclass
class KlipParams:
    k_klip: int
    exclusions : list[ExclusionValues]
    decomposer : Callable
    reuse : bool = False
    initial_decomposer : Optional[Callable] = None
    missing_data_value: float = np.nan
    strategy : constants.KlipStrategy = constants.KlipStrategy.DOWNDATE_SVD
    initial_decomposition : Optional[InitialDecomposition] = None
    warmup : bool = False

    def __post_init__(self):
        if self.initial_decomposer is None:
            self.initial_decomposer = self.decomposer
distributed.protocol.register_generic(KlipParams)

import logging

log = logging.getLogger(__name__)


def mean_subtract_vecs(image_vecs: np.ndarray):
    mean_vec = np.average(image_vecs, axis=1)
    image_vecs_meansub = image_vecs - mean_vec[:, np.newaxis]
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
        ref = utils.drop_idx_range_cols(
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
    image_vecs_meansub, mean_vec = mean_subtract_vecs(image_vecs)
    # since the klip implementation is going column-wise
    # this will make each column contiguous
    image_vecs_meansub = np.asfortranarray(image_vecs_meansub)
    if params.strategy in (constants.KlipStrategy.DOWNDATE_SVD, constants.KlipStrategy.SVD):
        return klip_mtx_svd(image_vecs_meansub, params), mean_vec
    elif params.strategy is constants.KlipStrategy.COVARIANCE:
        return klip_mtx_covariance(image_vecs_meansub, params), mean_vec
    else:
        raise ValueError(f"Unknown strategy value in {params=}")


def klip_mtx_covariance(image_vecs_meansub : np.ndarray, params : KlipParams):
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
    exclusion_values, exclusion_deltas = _exclusions_to_arrays(params)
    if params.reuse:
        core.set_num_mkl_threads(core.MKL_MAX_THREADS)
        lambda_values, mtx_c = params.decomposer(mtx_e_all, k_klip)
        eigenimages = image_vecs_meansub @ (mtx_c * np.power(lambda_values, -1/2))
    # TODO: parallelize covariance loop
    # # core.set_num_mkl_threads(1)
    for i in range(n_images):
        if not params.reuse:
            min_excluded_idx, max_excluded_idx = exclusions_to_range(
                n_images=n_images,
                current_idx=i,
                exclusion_values=exclusion_values,
                exclusion_deltas=exclusion_deltas
            )
            min_excluded_idx = max(min_excluded_idx, 0)
            max_excluded_idx = min(n_images, max_excluded_idx)
            mtx_e = utils.drop_idx_range_rows_cols(mtx_e_all, min_excluded_idx, max_excluded_idx + 1)
            lambda_values, mtx_c = params.decomposer(mtx_e, k_klip)
            ref_subset = np.hstack((image_vecs_meansub[:,:min_excluded_idx], image_vecs_meansub[:,max_excluded_idx+1:]))
            eigenimages = ref_subset @ (mtx_c * np.power(lambda_values, -1/2))

        meansub_target = image_vecs_meansub[:,i]
        output[:,i] = meansub_target - eigenimages @ (eigenimages.T @ meansub_target)
    return output

@njit(cache=True)
def get_excluded_mask(values, exclude_within_delta, current_value=None):
    deltas = np.abs(values - current_value)
    return deltas <= exclude_within_delta

# @njit(numba.boolean[:](
#     numba.intc, # n_images
#     numba.intc, # current_idx
#     numba.float32[:,:], # exclusion_values
#     numba.float32[:] # exclusion_deltas
# ))
@njit(cache=True)
def exclusions_to_mask(n_images, current_idx, exclusion_values, exclusion_deltas):
    mask = np.zeros(n_images, dtype=numba.boolean)
    mask[current_idx] = True  # exclude at least the current frame
    if exclusion_values is not None:
        for i in range(len(exclusion_values)):
            excl_values = exclusion_values[i]
            excl_delta = exclusion_deltas[i]
            individual_mask = get_excluded_mask(excl_values, excl_delta, excl_values[current_idx])
            mask |= individual_mask
    return mask

# @njit((
#     numba.intc, # n_images
#     numba.intc, # current_idx
#     numba.optional(numba.float32[:,:]), # exclusion_values
#     numba.optional(numba.float32[:]), # exclusion_deltas
# ))
@njit(cache=True)
def exclusions_to_range(n_images, current_idx, exclusion_values, exclusion_deltas):
    mask = exclusions_to_mask(n_images, current_idx, exclusion_values, exclusion_deltas)
    indices = np.argwhere(mask)
    min_excluded_idx = np.min(indices)
    max_excluded_idx = np.max(indices)
    # detect if this isn't a simple range (which we can't handle)
    frame_idxs = np.arange(n_images)
    contiguous_mask = (frame_idxs >= min_excluded_idx) & (frame_idxs <= max_excluded_idx)
    if np.count_nonzero(mask ^ contiguous_mask):
        raise ValueError("Non-contiguous ranges to exclude detected, but we don't handle that")
    return min_excluded_idx, max_excluded_idx

@njit(
    # TODO figure out the right signature to compile this at import time
    # (
    #     numba.float32[:,:],
    #     numba.intp,
    #     numba.float32[:,:],
    #     numba.float32[:],
    #     numba.float32[:,:],
    #     numba.intc,
    #     numba.boolean,
    #     numba.typeof(constants.KlipStrategy.SVD),
    #     numba.optional(numba.float32[:,:]),
    #     numba.optional(numba.float32[:,:])
    # ),
    parallel=True
)
def klip_chunk_svd(
    image_vecs_meansub, n_images, mtx_u0, diag_s0, mtx_v0, k_klip, reuse, strategy,
    exclusion_values, exclusion_deltas
):
    n_frames = image_vecs_meansub.shape[1]
    output = np.zeros_like(image_vecs_meansub)
    print('klip_chunk_svd running with', numba.get_num_threads(), 'threads on', n_frames, 'frames')
    for i in numba.prange(n_frames):
        if not reuse:
            min_excluded_idx, max_excluded_idx = exclusions_to_range(
                n_images=n_images,
                current_idx=i,
                exclusion_values=exclusion_values,
                exclusion_deltas=exclusion_deltas,
            )
            n_excluded = max_excluded_idx - min_excluded_idx + 1
            print('processing frame', i, ', excluding', n_excluded, ' frames (from frame', min_excluded_idx, 'to', max_excluded_idx, ")")
            if strategy == constants.KlipStrategy.DOWNDATE_SVD:
                assert mtx_u0 is not None
                assert diag_s0 is not None
                assert mtx_v0 is not None
                subset_mtx_u0 = np.ascontiguousarray(mtx_u0[:,:k_klip + n_excluded])
                subset_diag_s = diag_s0[:k_klip + n_excluded]
                subset_mtx_v0 = np.ascontiguousarray(mtx_v0[:,:k_klip + n_excluded])
                new_u, _, _ = learning.minimal_downdate(
                    subset_mtx_u0,
                    subset_diag_s,
                    subset_mtx_v0,
                    min_col_to_remove=min_excluded_idx,
                    max_col_to_remove=max_excluded_idx + 1,
                )
                eigenimages = new_u[:, :k_klip]
            else:
                subset_image_vecs = utils.drop_idx_range_cols(image_vecs_meansub, min_excluded_idx, max_excluded_idx + 1)
                eigenimages, _, _ = learning._numba_svd_wrap(subset_image_vecs, k_klip)
        else:
            assert mtx_u0 is not None
            eigenimages = mtx_u0[:, :k_klip]
        meansub_target = image_vecs_meansub[:, i]
        # Since we may have truncated by columns above, this re-contiguou-fies
        # and silences the NumbaPerformanceWarning
        eigenimages = np.ascontiguousarray(eigenimages)
        output[:, i] = meansub_target - eigenimages @ (eigenimages.T @ meansub_target)
    return output

def _exclusions_to_arrays(params):
    if not params.reuse and len(params.exclusions) > 0:
        exclusion_values = np.stack([excl.values for excl in params.exclusions])
        exclusion_deltas = np.stack([excl.exclude_within_delta for excl in params.exclusions])
    else:
        exclusion_values = exclusion_deltas = None
    return exclusion_values, exclusion_deltas

def klip_mtx_svd(image_vecs_meansub, params : KlipParams):
    k_klip = params.k_klip
    output = np.zeros_like(image_vecs_meansub)
    total_n_frames = image_vecs_meansub.shape[1]
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
        f"{image_vecs_meansub.shape=}, {k_klip=}, {initial_k=}, {image_vecs_meansub.flags=}"
    )
    if image_vecs_meansub.shape[0] < initial_k or image_vecs_meansub.shape[1] < initial_k:
        raise ValueError(f"Number of modes requested exceeds dimensions of input")
    
    if (
        params.strategy is constants.KlipStrategy.DOWNDATE_SVD or 
        (params.strategy is constants.KlipStrategy.SVD and params.reuse)
    ):
        initial_decomposition = params.initial_decomposition
        if initial_decomposition is None:
            # All hands on deck for initial decomposition
            core.set_num_mkl_threads(core.MKL_MAX_THREADS)
            log.info(f'Computing initial decomposition')
            mtx_u0, diag_s0, mtx_v0 = learning.generic_svd(image_vecs_meansub, initial_k)
            # Maximize number of independent subproblems
            core.set_num_mkl_threads(1)
            log.info(f"Done computing initial decomposition")
        else:
            mtx_u0 = np.ascontiguousarray(initial_decomposition.mtx_u0[:, :initial_k])
            diag_s0 = np.ascontiguousarray(initial_decomposition.diag_s0[:initial_k])
            mtx_v0 = np.ascontiguousarray(initial_decomposition.mtx_v0[:, :initial_k])
    else:
        mtx_u0 = diag_s0 = mtx_v0 = None

    if params.warmup:
        return InitialDecomposition(mtx_u0, diag_s0, mtx_v0, image_vecs_meansub)

    exclusion_values, exclusion_deltas = _exclusions_to_arrays(params)
    log.info(f'Computing KLIPed vectors')
    start = time.perf_counter()
    output = klip_chunk_svd(
        image_vecs_meansub,
        total_n_frames,
        mtx_u0,
        diag_s0,
        mtx_v0,
        k_klip,
        params.reuse,
        params.strategy,
        exclusion_values,
        exclusion_deltas
    )
    end = time.perf_counter()
    log.info(f"Finished KLIPing in {end - start}")
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
    constants.KlipStrategy.COVARIANCE: learning.eigh_top_k,
}
