import gc
from enum import Enum
import dask
from dataclasses import dataclass
from typing import Union, Optional
from collections.abc import Callable
import numpy as np
from scipy import sparse
import pylops
import pylops.optimization.solver
import dask.array as da
import distributed.protocol
from numba import njit
import numba
import time
import memory_profiler
from ..core import get_array_module
from ..core import cupy as cp
from .. import core, utils, constants
from . import learning, improc, characterization

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
class InitialDecomposition:
    mtx_u0 : np.ndarray
    diag_s0 : np.ndarray
    mtx_v0 : np.ndarray

    def __repr__(self):
        mtx_u0, diag_s0, mtx_v0 = self.mtx_u0, self.diag_s0, self.mtx_v0
        return f"InitialDecomposition<{mtx_u0.shape=}, {diag_s0.shape=}, {mtx_v0.shape=}>"


@dataclass
class KlipInput:
    sci_arr: np.ndarray
    obstable: np.ndarray
    estimation_mask: np.ndarray
    signal_arr : Optional[np.ndarray] = None

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
class KlipParams:
    k_klip: int
    exclusions : list[ExclusionValues]
    decomposer : Callable
    reuse : bool = False
    initial_decomposer : Optional[Callable] = None
    missing_data_value: float = np.nan
    strategy : constants.KlipStrategy = constants.KlipStrategy.DOWNDATE_SVD
    initial_decomposition : Optional[InitialDecomposition] = None
    initial_decomposition_only : bool = False

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


def klip_mtx(image_vecs, params : KlipParams, signal_vecs=None):
    image_vecs_meansub, mean_vec = mean_subtract_vecs(image_vecs)
    # since the klip implementation is going column-wise
    # this will make each column contiguous
    image_vecs_meansub = np.asfortranarray(image_vecs_meansub)
    if signal_vecs is not None:
        signal_vecs = np.asfortranarray(signal_vecs - mean_vec)
    if params.strategy in (constants.KlipStrategy.DOWNDATE_SVD, constants.KlipStrategy.SVD):
        return klip_mtx_svd(image_vecs_meansub, params, signal_vecs), mean_vec
    elif params.strategy is constants.KlipStrategy.COVARIANCE:
        return klip_mtx_covariance(image_vecs_meansub, params, signal_vecs), mean_vec
    else:
        raise ValueError(f"Unknown strategy value in {params=}")


def klip_mtx_covariance(image_vecs_meansub : np.ndarray, params : KlipParams, signal_vecs : np.ndarray):
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
    if signal_vecs is not None:
        output_model = np.zeros_like(image_vecs_meansub)
    else:
        output_model = None
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
        if signal_vecs is not None:
            model_target = signal_vecs[:,i]
            output_model[:,i] = model_target - eigenimages @ (eigenimages.T @ model_target)
    return output, output_model

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
    exclusion_values, exclusion_deltas, signal_vecs
):
    n_frames = image_vecs_meansub.shape[1]
    output = np.zeros_like(image_vecs_meansub)
    if signal_vecs is not None:
        output_model = np.zeros_like(signal_vecs)
    else:
        output_model = None
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
    return output, output_model

def _exclusions_to_arrays(params):
    if not params.reuse and len(params.exclusions) > 0:
        exclusion_values = np.stack([excl.values for excl in params.exclusions])
        exclusion_deltas = np.stack([excl.exclude_within_delta for excl in params.exclusions])
    else:
        exclusion_values = exclusion_deltas = None
    return exclusion_values, exclusion_deltas

def klip_mtx_svd(image_vecs_meansub, params : KlipParams, signal_vecs):
    k_klip = params.k_klip
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
            if mtx_u0.shape[0] != image_vecs_meansub.shape[0]:
                raise ValueError(f"Initial decomposition has {mtx_u0.shape=} but {image_vecs_meansub.shape=}. Mask changed?")
            if mtx_v0.shape[0] != image_vecs_meansub.shape[1]:
                raise ValueError(f"Initial decomposition has {mtx_v0.shape=} but {image_vecs_meansub.shape=}. Combination parameters changed?")
    else:
        mtx_u0 = diag_s0 = mtx_v0 = None

    if params.initial_decomposition_only:
        log.debug("Bailing out early to store decomposition")
        return InitialDecomposition(mtx_u0, diag_s0, mtx_v0)

    exclusion_values, exclusion_deltas = _exclusions_to_arrays(params)
    log.info(f'Computing KLIPed vectors')
    start = time.perf_counter()
    output, output_model = klip_chunk_svd(
        image_vecs_meansub,
        total_n_frames,
        mtx_u0,
        diag_s0,
        mtx_v0,
        k_klip,
        params.reuse,
        params.strategy,
        exclusion_values,
        exclusion_deltas,
        signal_vecs
    )
    end = time.perf_counter()
    log.info(f"Finished KLIPing in {end - start}")
    return output, output_model


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

@dataclass
class TrapBasis:
    temporal_basis : np.ndarray
    time_sec : float
    pix_used : int

@dataclass
class TrapParams:
    # modes_frac : float = 0.3
    k_modes : int
    model_trim_threshold : float = 0.2
    model_pix_threshold : float = 0.3
    compute_residuals : bool = True
    incorporate_offset : bool = True
    scale_ref_std : bool = True
    scale_model_std : bool = True
    dense_decomposer : callable = learning.generic_svd
    iterative_decomposer : callable = learning.cpu_top_k_svd_arpack
    min_modes_frac_for_dense : float = 0.15  # dense solver presumed faster when more than this fraction of all modes requested
    min_dim_for_iterative : int = 1000   # dense solver is as fast or faster below some matrix dimension
    force_gpu_decomposition : bool = False
    force_gpu_inversion : bool = False
    # arguments to pylops.optimization.solver.cgls
    damp : float = 1e-8
    tol : float = 1e-8
    # make it possible to pass in basis
    return_basis : bool = False
    precomputed_basis : Optional[TrapBasis] = None
    background_split_mask: Optional[np.ndarray] = None

def trap_mtx(image_vecs, model_vecs, trap_params : TrapParams):
    xp = core.get_array_module(image_vecs)
    was_gpu_array = xp is cp
    timers = {}
    model_threshold = trap_params.model_trim_threshold
    pix_below_threshold = model_vecs / xp.max(model_vecs, axis=0) < model_threshold
    trimmed_model_vecs = model_vecs.copy()
    trimmed_model_vecs[pix_below_threshold] = 0
    planet_signal_threshold = trap_params.model_pix_threshold
    median_timeseries = xp.median(image_vecs, axis=0)
    image_vecs_medsub = image_vecs - median_timeseries

    if trap_params.precomputed_basis is None:
        pix_with_planet_signal = xp.any(trimmed_model_vecs / xp.max(trimmed_model_vecs, axis=0),axis=1) > planet_signal_threshold
        pix_used = xp.count_nonzero(~pix_with_planet_signal)
        assert 0 < pix_used < trimmed_model_vecs.shape[0]
        ref_vecs = image_vecs_medsub[~pix_with_planet_signal]
        log.debug(f"Using {pix_used} pixel time series for TRAP basis with {trap_params.k_modes} modes")
        trap_basis = trap_phase_1(ref_vecs, trap_params)
    else:
        trap_basis = trap_phase_1(None, trap_params)
    if trap_params.return_basis and not was_gpu_array:
        if get_array_module(trap_basis.temporal_basis) is core.cupy:
            trap_basis.temporal_basis = trap_basis.temporal_basis.get()
        return trap_basis
    timers['time_svd_sec'] = trap_basis.time_sec
    if trap_params.force_gpu_inversion and not was_gpu_array:
        image_vecs_medsub_, trimmed_model_vecs_ = cp.asarray(image_vecs_medsub), cp.asarray(trimmed_model_vecs)
        del image_vecs_medsub, trimmed_model_vecs
        image_vecs_medsub, trimmed_model_vecs = image_vecs_medsub_, trimmed_model_vecs_
        temporal_basis = cp.asarray(trap_basis.temporal_basis)
    else:
        temporal_basis = trap_basis.temporal_basis
    model_coeff, inv_timers, maybe_resid_vecs = trap_phase_2(
        image_vecs_medsub, trimmed_model_vecs,
        temporal_basis, trap_params
    )
    timers.update(inv_timers)
    if trap_params.force_gpu_inversion and not was_gpu_array:
        if maybe_resid_vecs is not None:
            maybe_resid_vecs = maybe_resid_vecs.get()
        model_coeff = model_coeff.get()
    return model_coeff, timers, trap_basis.pix_used, maybe_resid_vecs

def trap_phase_1(ref_vecs, trap_params):
    if trap_params.precomputed_basis is not None:
        basis = trap_params.precomputed_basis
        temporal_basis = basis.temporal_basis
        xp = core.get_array_module(temporal_basis)
        temporal_basis = xp.ascontiguousarray(temporal_basis[:,:trap_params.k_modes])
        return TrapBasis(temporal_basis, basis.time_sec, basis.pix_used)
    xp = core.get_array_module(ref_vecs)
    # k_modes = int(min(image_vecs_medsub.shape) * trap_params.modes_frac)
    k_modes = trap_params.k_modes
    max_modes = min(ref_vecs.shape)

    # Using the std of each pixel's timeseries to scale it before decomposition reduces the weight of the brightest pixels
    if trap_params.scale_ref_std:
        ref_vecs_std = xp.std(ref_vecs, axis=1)
        scaled_ref_vecs = ref_vecs / ref_vecs_std[:,np.newaxis]
    else:
        scaled_ref_vecs = ref_vecs
    if trap_params.force_gpu_decomposition:
        scaled_ref_vecs = cp.asarray(scaled_ref_vecs)
        xp = core.cupy

    # select decomposer based on mode fraction and whether we're on GPU
    if xp is core.cupy:
        decomposer = trap_params.dense_decomposer
    elif k_modes > trap_params.min_modes_frac_for_dense * max_modes:
        decomposer = trap_params.dense_decomposer
    elif max_modes < trap_params.min_dim_for_iterative:
        decomposer = trap_params.dense_decomposer
    else:
        decomposer = trap_params.iterative_decomposer

    log.debug(f"Begin SVD with {decomposer=}...")
    time_sec = time.perf_counter()
    _, _, mtx_v = decomposer(scaled_ref_vecs, k_modes)
    time_sec = time.perf_counter() - time_sec
    log.debug(f"SVD complete in {time_sec} sec, constructing operator")
    temporal_basis = mtx_v  # shape = (nframes, ncomponents)
    if trap_params.return_basis and xp is not core.cupy:
        temporal_basis = temporal_basis.get()
    return TrapBasis(temporal_basis, time_sec, ref_vecs.shape[0])

def trap_phase_2(image_vecs_medsub, model_vecs, temporal_basis, trap_params):
    xp = core.get_array_module(image_vecs_medsub)
    was_gpu_array = xp is cp
    timers = {}
    flat_model_vecs = model_vecs.ravel()
    if trap_params.scale_model_std:
        model_coeff_scale = xp.std(flat_model_vecs)
        flat_model_vecs /= model_coeff_scale
    else:
        model_coeff_scale = 1
    if trap_params.force_gpu_inversion:
        temporal_basis = cp.asarray(temporal_basis)
        flat_model_vecs = cp.asarray(flat_model_vecs)
    operator_block_diag = [temporal_basis.T] * image_vecs_medsub.shape[0]
    opstack = [
        pylops.BlockDiag(operator_block_diag),
    ]
    if trap_params.incorporate_offset:
        if trap_params.background_split_mask is not None:
            left_mask_vec = trap_params.background_split_mask
            left_mask_megavec = np.repeat(left_mask_vec[:,np.newaxis], model_vecs.shape[1]).ravel()
            assert len(left_mask_megavec) == len(flat_model_vecs)
            left_mask_megavec = left_mask_megavec[np.newaxis,:].astype(flat_model_vecs.dtype)
            left_mask_megavec = left_mask_megavec - left_mask_megavec.mean()
            left_mask_megavec /= np.linalg.norm(left_mask_megavec)
            # "ones" for left side pixels -> fit constant offset for left psf
            opstack.append(xp.asarray(left_mask_megavec))
            # "ones" for right side pixels -> fit constant offset for right psf
            right_mask_megavec = -1 * left_mask_megavec
            opstack.append(xp.asarray(right_mask_megavec))
        else:
            background_megavec = np.ones_like(flat_model_vecs[xp.newaxis, :])
            background_megavec /= np.linalg.norm(background_megavec)
            opstack.append(xp.asarray(background_megavec))
    opstack.append(flat_model_vecs[xp.newaxis, :])
    op = pylops.VStack(opstack).transpose()
    log.debug(f"TRAP operator: {op}")

    image_megavec = image_vecs_medsub.ravel()
    solver_kwargs = dict(damp=trap_params.damp, tol=trap_params.tol)
    log.debug(f"Performing inversion on A.shape={op.shape} and b={image_megavec.shape}")
    timers['invert'] = time.perf_counter()
    solver = pylops.optimization.solver.cgls
    cgls_result = solver(
        op,
        image_megavec,
        xp.zeros(int(op.shape[1])),
        **solver_kwargs
    )
    xinv = cgls_result[0]
    timers['invert'] = time.perf_counter() - timers['invert']
    log.debug(f"Finished RegularizedInversion in {timers['invert']} sec")
    if core.get_array_module(xinv) is cp:
        model_coeff = float(xinv.get()[-1])
    else:
        model_coeff = float(xinv[-1])
    model_coeff = model_coeff / model_coeff_scale
    # return model_coeff, timers
    if trap_params.compute_residuals:
        solnvec = xinv
        solnvec[-1] = 0  # zero planet model contribution
        log.debug(f"Constructing starlight estimate from fit vector")
        timers['subtract'] = time.perf_counter()
        estimate_vecs = op.dot(solnvec).reshape(image_vecs_medsub.shape)
        if core.get_array_module(image_vecs_medsub) is not core.get_array_module(estimate_vecs):
            image_vecs_medsub = core.get_array_module(estimate_vecs).asarray(image_vecs_medsub)
        resid_vecs = image_vecs_medsub - estimate_vecs
        if core.get_array_module(resid_vecs) is cp and not was_gpu_array:
            resid_vecs = resid_vecs.get()
        timers['subtract'] = time.perf_counter() - timers['subtract']
        log.debug(f"Starlight subtracted in {timers['subtract']} sec")
        return model_coeff, timers, resid_vecs
    else:
        return model_coeff, timers, None


def trap_evaluate_point(
    r_px,
    pa_deg,
    data_cube,
    mask,
    psf,
    scale_factors,
    angles,
    inject=False,
    scale=1,
    modes_frac=0.3,
    use_gpu=False,
):
    all_start = start = time.perf_counter()
    log.debug("Injecting signals")
    scale_free_spec = characterization.CompanionSpec(r_px=r_px, pa_deg=pa_deg, scale=1)
    _discard_, signal_cube = characterization.inject_signals(
        data_cube, [scale_free_spec], psf, angles, scale_factors
    )
    del _discard_
    gc.collect()
    injected = data_cube + scale * signal_cube
    if inject and scale == 1:
        raise ValueError("!")
    log.debug(f"Injection done in {time.perf_counter() - start}")

    log.debug("Unwrapping cubes with mask")
    start = time.perf_counter()
    if inject:
        image_vecs = improc.unwrap_cube(injected, mask)
    else:
        image_vecs = improc.unwrap_cube(data_cube, mask)
    model_vecs = improc.unwrap_cube(signal_cube, mask)
    log.debug(f"Unwrapping done in {time.perf_counter() - start}")

    log.debug("TRAP++ing...")
    start = time.perf_counter()
    params = TrapParams(modes_frac=modes_frac)
    if use_gpu:
        gpu_image_vecs, gpu_model_vecs = cp.asarray(image_vecs), cp.asarray(model_vecs)
        del image_vecs, model_vecs
        image_vecs, model_vecs = gpu_image_vecs, gpu_model_vecs
    resid_vecs, model_coeff = trap_mtx(
        image_vecs, model_vecs, params
    )
    log.debug(f"TRAP++ing done in {time.perf_counter() - start}")
    del resid_vecs, image_vecs, model_vecs
    gc.collect()
    log.debug(f"Evaluated point in {time.perf_counter() - all_start}")
    return model_coeff
