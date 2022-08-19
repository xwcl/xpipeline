from collections import defaultdict
import orjson
from pprint import pformat
from astropy.convolution import convolve_fft, Tophat2DKernel
import math
from astropy.io import fits
import time
from copy import copy, deepcopy
from dataclasses import dataclass
import dataclasses
import xconf
import numpy as np
from xconf.contrib import BaseRayGrid, FileConfig, join, PathConfig, DirectoryConfig
import sys
import logging
from typing import Optional, Union, ClassVar
from ..commands.base import BaseCommand
from .. import utils
from ..tasks import starlight_subtraction, learning, improc, characterization
from ..tasks.characterization import CompanionSpec, r_pa_to_x_y, snr_from_convolution, calculate_snr
from .. import constants
from xpipeline.types import FITS_EXT
import logging

log = logging.getLogger(__name__)

POST_FILTER_NAMES = ('tophat', 'matched')
DEFAULT_DESTINATION_EXTS_FACTORY = lambda: ['finim']

@xconf.config
class PixelRotationRangeConfig:
    delta_px : float = xconf.field(default=0, help="Maximum difference between target frame value and matching frames")
    r_px : float = xconf.field(default=None, help="Radius at which to calculate motion in pixels")

@xconf.config
class AngleRangeConfig:
    delta_deg : float = xconf.field(default=0, help="Maximum difference between target frame value and matching frames")


@xconf.config
class FileConfig:
    path : str = xconf.field(help="File path")

    def open(self, mode='rb'):
        from ..utils import get_fs
        fs = get_fs(self.path)
        return fs.open(self.path, mode)

@xconf.config
class FitsConfig(FileConfig):
    path : str = xconf.field(help="Path from which to load the containing FITS file")
    ext : Union[int,str] = xconf.field(default=0, help="Extension from which to load")
    _cache : ClassVar[dict] = {}

    def _load_hdul(self, cache):
        from ..tasks import iofits
        if cache and self.path in self._cache:
            return self._cache[self.path]
        with self.open() as fh:
            hdul = iofits.load_fits(fh)
        for ext in hdul.extnames:
            hdul[ext].data.flags.writeable = False
        if cache:
            self._cache[self.path] = hdul
        return hdul

    def load(self, cache=True) -> np.ndarray:
        hdul = self._load_hdul(cache)
        data = hdul[self.ext].data
        return data

class PreloadedArray:
    array : np.ndarray
    def __init__(self, array):
        self.array = array
    def load(self) -> np.ndarray:
        return self.array
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.array.shape} [{self.array.dtype}]>"

@xconf.config
class FitsTableColumnConfig(FitsConfig):
    table_column : str = xconf.field(help="Path from which to load the containing FITS file")
    ext : Union[int,str] = xconf.field(default="OBSTABLE", help="Extension from which to load")

    def load(self, cache=True) -> np.ndarray:
        hdul = self._load_hdul(cache)
        coldata = hdul[self.ext].data[self.table_column]
        return coldata

@xconf.config
class Pipeline:
    def execute(self):
        """Returns result of pipeline given inputs in config
        """
        raise NotImplementedError("Subclasses must implement the execute() method")

@xconf.config
class SamplingConfig:
    n_radii : int = xconf.field(help="Number of steps in radius at which to probe contrast")
    spacing_px : float = xconf.field(help="Spacing in pixels between contrast probes along circle (sets number of probes at radius by 2 * pi * r / spacing)")
    scales : list[float] = xconf.field(default_factory=lambda: [0.0], help="Probe contrast levels (C = companion / host)")
    iwa_px : float = xconf.field(help="Inner working angle (px)")
    owa_px : float = xconf.field(help="Outer working angle (px)")

@xconf.config
class RadialMaskConfig:
    min_r_px : Union[int, float, None] = xconf.field(default=None, help="Apply radial mask excluding pixels < mask_min_r_px from center")
    max_r_px : Union[int, float, None] = xconf.field(default=None, help="Apply radial mask excluding pixels > mask_max_r_px from center")

    def get_mask(self, arr_or_shape):
        if hasattr(arr_or_shape, 'shape'):
            shape = arr_or_shape.shape
        else:
            shape = arr_or_shape
        from ..tasks import improc
        rho, _ = improc.polar_coords(improc.arr_center(shape), shape)
        mask = np.ones(shape, dtype=bool)
        if self.min_r_px is not None:
            mask &= (rho > self.min_r_px)
        if self.max_r_px is not None:
            mask &= (rho < self.max_r_px)
        return mask

@dataclass
class ModelSignalInput:
    arr: np.ndarray
    scale_factors: np.ndarray

@xconf.config
class ModelSignalInputConfig:
    model : FitsConfig = xconf.field(help="Model signal analogous to single science cube frame")
    scale_factors : Union[FitsConfig, FitsTableColumnConfig, None] = xconf.field(help="1-D array or table column of scale factors that make the amplitude of the model_arr signal match that of the primary")

    def load(self) -> ModelSignalInput:
        return ModelSignalInput(
            arr=self.model.load(),
            scale_factors=self.scale_factors.load() if self.scale_factors is not None else None
        )

@dataclass
class PipelineInput:
    sci_arr: np.ndarray
    estimation_mask: np.ndarray
    destination_exts: list[str] = dataclasses.field(default_factory=DEFAULT_DESTINATION_EXTS_FACTORY)
    combination_mask: Optional[np.ndarray] = None
    model_inputs : Optional[ModelSignalInput] = None
    model_arr : Optional[np.ndarray] = None

@dataclass
class StarlightSubtractionData:
    inputs : list[PipelineInput]
    angles : np.ndarray
    times_sec : Optional[np.ndarray]
    companions : list[CompanionSpec]

    def from_slice(self, the_slice):
        new_inputs = []
        for pi in self.inputs:
            new_inputs.append(PipelineInput(
                sci_arr=pi.sci_arr[the_slice],
                model_arr=pi.model_arr[the_slice],
                destination_exts=pi.destination_exts,
                estimation_mask=pi.estimation_mask,
                combination_mask=pi.combination_mask,
                model_inputs=pi.model_inputs
            ))
        return self.__class__(
            inputs=new_inputs,
            angles=self.angles[the_slice],
            times_sec=self.times_sec[the_slice],
            companions=self.companions,
        )

    def observations_count(self):
        return self.inputs[0].sci_arr.shape[0]

    def max_rank(self):
        cols = 0
        rows = 0
        for pi in self.inputs:
            if cols == 0:
                cols = pi.sci_arr.shape[0]
            else:
                if pi.sci_arr.shape[0] != cols:
                    raise ValueError("Inconsistent number of observations per input")
            rows += np.count_nonzero(pi.estimation_mask)
        return min(rows, cols)



@dataclass
class PipelineOutput:
    sci_arr: np.ndarray
    destination_exts: list[str] = dataclasses.field(default_factory=DEFAULT_DESTINATION_EXTS_FACTORY)
    model_arr: Optional[np.ndarray] = None
    mean_image: Optional[np.ndarray] = None

@xconf.config
class PipelineInputConfig:
    sci_arr: FitsConfig = xconf.field(help="Science frames as cube")
    estimation_mask: Optional[FitsConfig] = xconf.field(default=None, help="Estimation mask with the shape of a single plane of sci_arr")
    combination_mask: Optional[FitsConfig] = xconf.field(default=None, help="Combination mask with the shape of a single plane of sci_arr, or False to exclude from final combination")
    radial_mask : Optional[RadialMaskConfig] = xconf.field(default=None, help="Radial mask to exclude pixels min_r_px > r || r > max_r_px from center")
    model_inputs : Optional[ModelSignalInputConfig] = xconf.field(default=None, help="Model signal for matched filtering")
    destination_exts: list[str] = xconf.field(default_factory=DEFAULT_DESTINATION_EXTS_FACTORY, help="Extension(s) into which final image should be combined")

    def load(self) -> PipelineInput:
        sci_arr = self.sci_arr.load()
        estimation_mask, combination_mask = self.get_masks(sci_arr.shape[1:])
        model_inputs = None
        if self.model_inputs is not None:
            model_inputs = self.model_inputs.load()
        else:
            model_inputs = None
        return PipelineInput(
            sci_arr=sci_arr,
            estimation_mask=estimation_mask,
            combination_mask=combination_mask,
            model_inputs=model_inputs,
            destination_exts=self.destination_exts,
        )

    def get_masks(self, single_plane_shape):
        if self.estimation_mask is not None:
            estimation_mask = self.estimation_mask.load()
        else:
            estimation_mask = np.ones(single_plane_shape, dtype=bool)
        if self.combination_mask is not None:
            combination_mask = self.combination_mask.load()
            if combination_mask.shape != estimation_mask.shape:
                raise ValueError(f"{combination_mask.shape=} != {estimation_mask.shape=}")
            combination_mask = combination_mask & estimation_mask
        else:
            combination_mask = estimation_mask
        if self.radial_mask is not None:
            radial_mask = self.radial_mask.get_mask(estimation_mask)
            combination_mask = combination_mask & radial_mask
            estimation_mask = estimation_mask & radial_mask
        return (
            estimation_mask,
            combination_mask,
        )

@xconf.config
class MeasurementConfig:
    r_px : float = xconf.field(help="Radius of companion")
    pa_deg : float = xconf.field(help="Position angle of companion in degrees East of North")

@xconf.config
class CompanionConfig(MeasurementConfig):
    scale : float = xconf.field(help=utils.unwrap(
        """Scale factor multiplied by template (and optional template
        per-frame scale factor) to give companion image,
        i.e., contrast ratio. Can be negative or zero."""))
    def to_companionspec(self):
        from xpipeline.tasks.characterization import CompanionSpec
        return CompanionSpec(self.r_px, self.pa_deg, self.scale)

@dataclass
class KlipTFmPointResult:
    snr : float
    signal : float
    image : np.ndarray
    filtered_image : np.ndarray
    matched_filter : np.ndarray


# def generate_model(model_inputs : ModelInputs, companion_r_px, companion_pa_deg):
#     companion_spec = characterization.CompanionSpec(companion_r_px, companion_pa_deg, 1.0)
#     # generate
#     left_model_cube = characterization.generate_signals(
#         model_inputs.data_cube_shape,
#         [companion_spec],
#         model_inputs.left_template,
#         model_inputs.angles,
#         model_inputs.left_scales
#     )
#     right_model_cube = characterization.generate_signals(
#         model_inputs.data_cube_shape,
#         [companion_spec],
#         model_inputs.right_template,
#         model_inputs.angles,
#         model_inputs.right_scales
#     )
#     # stitch
#     out_cube = pipelines.vapp_stitch(left_model_cube, right_model_cube, clio.VAPP_PSF_ROTATION_DEG)
#     model_vecs = improc.unwrap_cube(out_cube, model_inputs.mask)
#     return model_vecs

from ..tasks.characterization import generate_signal, inject_signals

# from copy import deepcopy

# @xconf.config
# class MultipleCompanionsConfig:
#     injected_companions : list[CompanionConfig] = xconf.field(default_factory=list, help="Companions to inject")
#     probe_location : MeasurementConfig = xconf.field(help="Where to evaluate the SNR")

@xconf.config
class ExcludeRangeConfig:
    angle : Union[AngleRangeConfig,PixelRotationRangeConfig] = xconf.field(default=AngleRangeConfig(), help="Apply exclusion to derotation angles")
    nearest_n_frames : int = xconf.field(default=0, help="Number of additional temporally-adjacent frames on either side of the target frame to exclude from the sequence when computing the KLIP eigenimages")

@xconf.config
class Klip:
    klip : bool = xconf.field(default=True, help="Include this option to explicitly select the Klip strategy")
    return_basis : bool = xconf.field(default=False, help="Bail out early and return the basis set")
    reuse : bool = xconf.field(default=False, help="Use the same basis set for all frames")
    exclude : ExcludeRangeConfig = xconf.field(default=ExcludeRangeConfig(), help="How to exclude frames from reference sample")
    decomposer : learning.Decomposers = xconf.field(default=learning.Decomposers.svd, help="Modal decomposer for data matrix")

    def _make_exclusions(self, exclude : ExcludeRangeConfig, derotation_angles):
        exclusions = []
        if exclude.nearest_n_frames > 0:
            indices = np.arange(derotation_angles.shape[0])
            exc = starlight_subtraction.ExclusionValues(
                exclude_within_delta=exclude.nearest_n_frames,
                values=indices,
                num_excluded_max=2 * exclude.nearest_n_frames + 1
            )
            exclusions.append(exc)
        if isinstance(exclude.angle, PixelRotationRangeConfig) and exclude.angle.delta_px > 0:
            exc = starlight_subtraction.ExclusionValues(
                exclude_within_delta=exclude.angle.delta_px,
                values=exclude.angle.r_px * np.unwrap(np.deg2rad(derotation_angles))
            )
            exclusions.append(exc)
        elif isinstance(exclude.angle, AngleRangeConfig) and exclude.angle.delta_deg > 0:
            exc = starlight_subtraction.ExclusionValues(
                exclude_within_delta=exclude.angle.delta_deg,
                values=derotation_angles
            )
            exclusions.append(exc)
        else:
            pass  # not an error to have delta of zero, just don't exclude based on rotation
        log.debug(f"{exclusions=}")
        return exclusions

    def prepare(
        self,
        data : StarlightSubtractionData,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        decomposition: Optional[learning.PrecomputedDecomposition] = None,
    ):
        assert decomposition is None
        params = starlight_subtraction.KlipParams(
            k_klip=k_modes,
            exclusions=self._make_exclusions(self.exclude, angles),
            decomposer=self.decomposer.to_callable(),
            initial_decomposition_only=True,
            reuse=self.reuse,
        )
        image_vecs, probe_model_vecs = unwrap_inputs_to_matrices(data.inputs)
        _, _, decomposition, _ = starlight_subtraction.klip_mtx(image_vecs, params=params)
        return decomposition


    def execute(
        self,
        data : StarlightSubtractionData,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        decomposition: Optional[learning.PrecomputedDecomposition] = None,
    ):
        params = starlight_subtraction.KlipParams(
            k_klip=k_modes,
            exclusions=self._make_exclusions(self.exclude, angles),
            decomposer=self.decomposer.to_callable(),
            initial_decomposition=decomposition,
            reuse=self.reuse,
        )
        image_vecs, probe_model_vecs = unwrap_inputs_to_matrices(data.inputs)
        return starlight_subtraction.klip_mtx(image_vecs, params=params, probe_model_vecs=probe_model_vecs) + (angles,)

from numba import njit
@njit
def construct_time_contiguous_pairs(vectors, angles, times_sec):
    dt = 5 * np.median(np.diff(times_sec))
    vector_pairs = np.zeros((2 * vectors.shape[0], vectors.shape[1]), dtype=vectors.dtype)
    # print("before", vector_pairs.shape)
    out_angles = np.zeros((2, vectors.shape[1]))
    # print("before", out_angles.shape)
    col_cursor = 0
    for i in range(vectors.shape[1] - 1):
        delta = times_sec[i + 1] - times_sec[i]
        if delta > dt:
            # this is a chunk boundary, don't make a pair out of this index and i+1
            # print('chunk boundary at idx', i, 'delta was', delta, 'dt was', dt)
            continue
        vector_pairs[:, col_cursor] = np.concatenate((vectors[:, i], vectors[:, i+1]))
        out_angles[0, col_cursor] = angles[i]
        out_angles[1, col_cursor] = angles[i + 1]
        col_cursor += 1
        # print(i, i+1, vector_pairs.shape, out_angles.shape, col_cursor)
    vector_pairs = vector_pairs[:, :col_cursor + 1]
    # print("after", vector_pairs.shape)
    # print("before2", out_angles.shape)
    out_angles = out_angles[:, :col_cursor+1]
    # print("after", out_angles.shape)
    return vector_pairs, out_angles


@xconf.config
class DynamicModeDecomposition:
    dynamic_mode_decomposition : bool = xconf.field(default=True, help="")
    do_train_test_split : bool = xconf.field(default=True, help="Whether to use half the pairs of observation vectors for training and the other half for estimation")
    reduce_both_halves : bool = xconf.field(default=False, help="Whether to repeat the procedure on the training set with the testing set, using all input data")
    enforce_time_contiguity : bool = xconf.field(default=False, help="Whether to check timestamps when making vector pairs for train and test")
    scale_by_pix_std : bool = xconf.field(default=False, help="Whether to scale the values in each pixel series by their stddev before decomposition, and back after subtraction (makes things worse, apparently)")
    scale_by_frame_std : bool = xconf.field(default=False, help="Whether to scale the pixel values in each frame by the frame's stddev before decomposition, and back after subtraction (doesn't make much difference, apparently)")
    use_iterative_svd : bool = xconf.field(default=False, help="For low numbers of modes it may be much faster to use an iterative SVD solver")
    truncate_before_mode_construction : bool = xconf.field(default=True, help="Whether to truncate the SVD before using it to construct the modal basis Phi (true), or merely truncate the final modal basis Phi (false)")

    def train_test_split(self, vectors, angles, times_sec):
        if self.enforce_time_contiguity:
            vec_pairs, angle_pairs = construct_time_contiguous_pairs(
                vectors,
                angles,
                np.ascontiguousarray(times_sec),
            )
            log.debug(f"Made {vec_pairs.shape=} pairs of vectors adjacent in time from input {vectors.shape=}")
        else:
            if vectors.shape[1] % 2 != 0:
                vectors = vectors[:,:-1]
                angles = angles[:-1]
            vec_pairs = utils.wrap_columns(vectors)
            log.debug(f"Interleaving column vectors gives {vec_pairs.shape=} pairs of vectors")
            # drop half the angles as well
            angle_pairs = utils.wrap_columns(angles[np.newaxis, :])

        train_vecs = utils.unwrap_columns(vec_pairs[:,::2])
        test_vecs = utils.unwrap_columns(vec_pairs[:,1::2])

        # assert np.all(train_vecs[:,0] == vectors[:,0])
        # assert np.all(train_vecs[:,1] == vectors[:,1])
        # assert np.all(train_vecs[:,2] == vectors[:,4])
        train_angles = utils.unwrap_columns(angle_pairs[:,::2]).flatten()
        test_angles = utils.unwrap_columns(angle_pairs[:,1::2]).flatten()

        return train_vecs, test_vecs, train_angles, test_angles

    def collect_inputs(self, data, angles):
        image_vecs, probe_model_vecs = unwrap_inputs_to_matrices(data.inputs)
        median_vec = np.median(image_vecs, axis=0)
        image_vecs_sub = image_vecs - median_vec

        if self.do_train_test_split:
            train_vecs, test_vecs, train_angles, test_angles = self.train_test_split(image_vecs_sub, angles, data.times_sec)
            train_probe_model_vecs, test_probe_model_vecs, _, _ = self.train_test_split(probe_model_vecs, angles, data.times_sec)
        else:
            vec_pairs, angle_pairs = construct_time_contiguous_pairs(
                image_vecs_sub,
                angles,
                np.ascontiguousarray(data.times_sec),
            )
            probe_model_vec_pairs, _ = construct_time_contiguous_pairs(
                probe_model_vecs,
                angles,
                np.ascontiguousarray(data.times_sec),
            )
            image_vecs_sub = utils.unwrap_columns(vec_pairs)
            probe_model_vecs = utils.unwrap_columns(probe_model_vec_pairs)
            angles = utils.unwrap_columns(angle_pairs).flatten()
            print(f"{image_vecs_sub.shape=} {angles.shape=}")
            train_vecs = test_vecs = image_vecs_sub
            train_probe_model_vecs = test_probe_model_vecs = probe_model_vecs
            train_angles = test_angles = angles

        vecs_std = train_vecs_std = test_vecs_std = None
        if self.scale_by_pix_std:
            if self.do_train_test_split:
                train_vecs_std = np.std(train_vecs, axis=1)
                train_vecs /= train_vecs_std[:, np.newaxis]
                train_probe_model_vecs /= train_vecs_std[:, np.newaxis]

                test_vecs_std = np.std(test_vecs, axis=1)
                test_vecs /= test_vecs_std[:, np.newaxis]
                test_probe_model_vecs /= test_vecs_std[:, np.newaxis]
            else:
                vecs_std = np.std(image_vecs_sub, axis=1)
                image_vecs_sub /= vecs_std[:, np.newaxis]
                probe_model_vecs /= vecs_std[:, np.newaxis]

        if self.scale_by_frame_std:
            if self.do_train_test_split:
                train_vecs_std = np.std(train_vecs, axis=0)
                train_vecs /= train_vecs_std[np.newaxis, :]
                train_probe_model_vecs /= train_vecs_std[np.newaxis, :]

                test_vecs_std = np.std(test_vecs, axis=0)
                test_vecs /= test_vecs_std[np.newaxis, :]
                test_probe_model_vecs /= test_vecs_std[np.newaxis, :]
            else:
                vecs_std = np.std(image_vecs_sub, axis=0)
                image_vecs_sub /= vecs_std[np.newaxis, :]
                probe_model_vecs /= test_vecs_std[np.newaxis, :]

        if self.do_train_test_split:
            partitions = [
                (train_vecs, train_probe_model_vecs, train_vecs_std, train_angles),
                (test_vecs, test_probe_model_vecs, test_vecs_std, test_angles),
            ]
        else:
            partitions = [(image_vecs_sub, probe_model_vecs, vecs_std, angles)]

        return (
            partitions,
            median_vec,
        )

    def prepare(self,
        data : StarlightSubtractionData,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        decomposition: Optional[learning.PrecomputedDecomposition] = None,
    ) -> list[learning.PrecomputedDecomposition]:
        solver = learning.generic_svd
        if self.use_iterative_svd:
            solver = learning.cpu_top_k_svd_arpack
        partitions, median_vec = self.collect_inputs(data, angles)
        decomps = []
        for idx, (vecs, model_vecs, vecs_std, part_angles) in enumerate(partitions):
            # print(f"{vecs.shape=} {part_angles.shape=}")
            if self.do_train_test_split and idx == 0:
                train_vecs = partitions[1][0]
            else:
                train_vecs = partitions[0][0]
                if self.do_train_test_split and not self.reduce_both_halves:
                    continue
            mtx_x = train_vecs[:,:-1]  # drop last column (time-step)
            # mtx_xprime = train_vecs[:,1:]  # drop first column (time-step)
            mtx_u, diag_s, mtx_v = solver(mtx_x, k_modes)
            decomps.append(learning.PrecomputedDecomposition(
                mtx_u0=mtx_u,
                diag_s0=diag_s,
                mtx_v0=mtx_v,
            ))
        return decomps
        train_vecs, _, _, _, _, _ = self.collect_inputs(data, angles)
        mtx_x = train_vecs[:,:-1]  # drop last column (time-step)
        solver = learning.generic_svd
        if self.use_iterative_svd:
            solver = learning.cpu_top_k_svd_arpack
        mtx_u, diag_s, mtx_v = solver(mtx_x, k_modes)
        return learning.PrecomputedDecomposition(
            mtx_u0=mtx_u,
            diag_s0=diag_s,
            mtx_v0=mtx_v,
        )

    def execute(
        self,
        data : StarlightSubtractionData,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        decomposition: Optional[list[learning.PrecomputedDecomposition]] = None,
    ):
        # if decomposition is None:
        #     decomposition = self.prepare(data, k_modes, angles=angles)
        # mtx_u, diag_s, mtx_v = decomposition.mtx_u0, decomposition.diag_s0, decomposition.mtx_v0
        partitions, median_vec = self.collect_inputs(data, angles)
        subtracted_vecs = None
        subtracted_model_vecs = None
        if decomposition is None:
            decomposition = self.prepare(data, k_modes, angles=angles)
        final_angles = None

        for idx, (vecs, model_vecs, vecs_std, part_angles) in enumerate(partitions):
            # print(f"{vecs.shape=} {part_angles.shape=}")
            if self.do_train_test_split and idx == 0:
                train_vecs = partitions[1][0]
            else:
                train_vecs = partitions[0][0]
                if self.do_train_test_split and not self.reduce_both_halves:
                    continue

            mtx_xprime = train_vecs[:,1:]  # drop first column (time-step)
            mtx_u, diag_s, mtx_v = decomposition[idx].mtx_u0, decomposition[idx].diag_s0, decomposition[idx].mtx_v0
            # print(f"{mtx_x.shape=} {mtx_xprime.shape=}")
            if self.truncate_before_mode_construction:
                mtx_u, diag_s, mtx_v = mtx_u[:,:k_modes], diag_s[:k_modes], mtx_v[:,:k_modes]
            mtx_s_inv = np.diag(diag_s**-1)
            mtx_atilde = mtx_u.T @ mtx_xprime @ mtx_v @ mtx_s_inv
            diag_eigs, mtx_w = np.linalg.eig(mtx_atilde)
            mtx_phi = mtx_xprime @ mtx_v @ mtx_s_inv @ mtx_w

            if not self.truncate_before_mode_construction:
                mtx_phi = mtx_phi[:,:k_modes]
                mtx_eigs = np.diag(diag_eigs[:k_modes])
                mtx_phi_pinv = np.linalg.pinv(mtx_phi)
            else:
                mtx_eigs = np.diag(diag_eigs)
                mtx_phi_pinv = np.linalg.pinv(mtx_phi)
            dmd_recons_vecs = (mtx_phi @ (mtx_eigs @ (mtx_phi_pinv @ vecs)))
            vecs = vecs - dmd_recons_vecs.real # intentionally making a copy because we need them for the other partition
            if self.scale_by_pix_std:
                vecs *= vecs_std[:, np.newaxis]  # train and test both scaled by std above, scale back
            if self.scale_by_frame_std:
                vecs *= vecs_std[np.newaxis, :]  # train and test both scaled by std above, scale back
            dmd_recons_probes = (mtx_phi @ (mtx_eigs @ (mtx_phi_pinv @ model_vecs)))
            probe_model_vecs_sub = model_vecs - dmd_recons_probes.real
            if self.scale_by_frame_std:
                probe_model_vecs_sub *= vecs_std[np.newaxis, :]  # scaled by std of test vecs above, scale back
            if self.scale_by_pix_std:
                probe_model_vecs_sub *= vecs_std[:, np.newaxis]  # scaled by std of test vecs above, scale back
            if subtracted_vecs is None:
                subtracted_vecs = vecs
                subtracted_model_vecs = probe_model_vecs_sub
                final_angles = part_angles
            else:
                # print(f"concatenate {subtracted_vecs.shape=} {vecs.shape=}")
                subtracted_vecs = np.concatenate([subtracted_vecs, vecs], axis=1)
                subtracted_model_vecs = np.concatenate([subtracted_model_vecs, probe_model_vecs_sub], axis=1)
                # print(f"concatenate {final_angles.shape=} {part_angles.shape=}")
                final_angles = np.concatenate([final_angles, part_angles])

        # print(f"{subtracted_vecs.shape=} {final_angles.shape=}")
        return subtracted_vecs, subtracted_model_vecs, decomposition, median_vec, final_angles

@xconf.config
class KlipSubspace:
    klip_subspace : bool = xconf.field(default=True, help="")
    model_trim_threshold : float = xconf.field(default=0.2, help="fraction of peak model intensity in a frame below which model is trimmed to zero")
    model_pix_threshold : float = xconf.field(default=0.3, help="max level in model pix for data pix to be included in ref vecs")
    scale_ref_std : bool = xconf.field(default=True, help="")
    scale_model_std : bool = xconf.field(default=True, help="")
    dense_decomposer : learning.Decomposers = xconf.field(default=learning.Decomposers.svd, help="Modal decomposer for data matrix when > (min_modes_frac_for_dense * n_obs) modes are requested, and dense subproblems")
    iterative_decomposer : learning.Decomposers = xconf.field(default=learning.Decomposers.svd_top_k, help="Modal decomposer when < (min_modes_frac_for_dense * n_obs) modes are requested")
    min_modes_frac_for_dense : float = xconf.field(default=0.15, help="Dense solver presumed faster when more than this fraction of all modes requested")
    min_dim_for_iterative : int = xconf.field(default=1000, help="Dense solver is as fast or faster below some matrix dimension so fall back to it")

    def construct_klipt_params_dict(self):
        return {
            'model_trim_threshold': self.model_trim_threshold,
            'model_pix_threshold': self.model_pix_threshold,
            'scale_ref_std': self.scale_ref_std,
            'dense_decomposer': self.dense_decomposer.to_callable(),
            'iterative_decomposer': self.iterative_decomposer.to_callable(),
            'min_modes_frac_for_dense': self.min_modes_frac_for_dense,
            'min_dim_for_iterative': self.min_dim_for_iterative,
        }

    def prepare(
        self,
        data : StarlightSubtractionData,
        # image_vecs: np.ndarray,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        decomposition: Optional[learning.PrecomputedDecomposition] = None,
    ) -> learning.PrecomputedDecomposition:
        image_vecs, probe_model_vecs = unwrap_inputs_to_matrices(data.inputs)
        image_vecs_medsub = image_vecs - np.median(image_vecs, axis=0)
        mtx_u, diag_s, mtx_v = learning.generic_svd(image_vecs_medsub, k_modes)
        return learning.PrecomputedDecomposition(mtx_u, diag_s, mtx_v)

    def execute(
        self,
        data : StarlightSubtractionData,
        # image_vecs: np.ndarray,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        decomposition: Optional[learning.PrecomputedDecomposition] = None,
    ):
        image_vecs, probe_model_vecs = unwrap_inputs_to_matrices(data.inputs)
        image_vecs_medsub = image_vecs - np.median(image_vecs, axis=0)
        mtx_u = decomposition.mtx_u0[:,:k_modes]
        diag_s = decomposition.diag_s0[:k_modes]
        mtx_v = decomposition.mtx_v0[:,:k_modes]
        subspace_image_vec_projections = (mtx_u * diag_s) @ mtx_v.T
        return image_vecs_medsub - subspace_image_vec_projections, probe_model_vecs, None, None, angles

@xconf.config
class KlipTranspose:
    klip_transpose : bool = xconf.field(default=True, help="Include this option to explicitly select the KlipTranspose strategy")
    model_trim_threshold : float = xconf.field(default=0.2, help="fraction of peak model intensity in a frame below which model is trimmed to zero")
    model_pix_threshold : float = xconf.field(default=0.3, help="max level in model pix for data pix to be included in ref vecs")
    scale_ref_std : bool = xconf.field(default=True, help="")
    scale_model_std : bool = xconf.field(default=True, help="")
    dense_decomposer : learning.Decomposers = xconf.field(default=learning.Decomposers.svd, help="Modal decomposer for data matrix when > (min_modes_frac_for_dense * n_obs) modes are requested, and dense subproblems")
    iterative_decomposer : learning.Decomposers = xconf.field(default=learning.Decomposers.svd_top_k, help="Modal decomposer when < (min_modes_frac_for_dense * n_obs) modes are requested")
    min_modes_frac_for_dense : float = xconf.field(default=0.15, help="Dense solver presumed faster when more than this fraction of all modes requested")
    min_dim_for_iterative : int = xconf.field(default=1000, help="Dense solver is as fast or faster below some matrix dimension so fall back to it")
    # make it possible to pass in basis
    return_basis : bool = xconf.field(default=False, help="Bail out early and return the temporal basis set")
    excluded_annulus_width_px : Optional[float] = xconf.field(default=None, help="Width of a mask annulus excluding pixels from the reference timeseries")

    def construct_klipt_params_dict(self):
        return {
            'model_trim_threshold': self.model_trim_threshold,
            'model_pix_threshold': self.model_pix_threshold,
            'scale_ref_std': self.scale_ref_std,
            'dense_decomposer': self.dense_decomposer.to_callable(),
            'iterative_decomposer': self.iterative_decomposer.to_callable(),
            'min_modes_frac_for_dense': self.min_modes_frac_for_dense,
            'min_dim_for_iterative': self.min_dim_for_iterative,
        }

    def prepare(
        self,
        data : StarlightSubtractionData,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        decomposition: Optional[learning.PrecomputedDecomposition] = None,
    ) -> learning.PrecomputedDecomposition:
        klipt_params = starlight_subtraction.KlipTParams(
            k_modes=k_modes,
            **self.construct_klipt_params_dict()
        )
        image_vecs, probe_model_vecs = unwrap_inputs_to_matrices(data.inputs)
        image_vecs_medsub = image_vecs - np.median(image_vecs, axis=0)
        mask_vec_chunks = []
        for pi in data.inputs:
            mask = np.zeros(pi.estimation_mask.shape, dtype=bool)
            if self.excluded_annulus_width_px is not None:
                rho, _ = improc.polar_coords(improc.arr_center(pi.estimation_mask), pi.estimation_mask.shape)
                for companion in data.companions:
                    mask |= np.abs(rho - companion.r_px) < self.excluded_annulus_width_px
            mask_vec_chunks.append(improc.unwrap_image(mask, pi.estimation_mask))
        excluded_ref_vecs = np.concatenate(mask_vec_chunks)
        assert excluded_ref_vecs.shape[0] == image_vecs.shape[0]
        return starlight_subtraction.compute_klipt_basis(image_vecs_medsub, probe_model_vecs, klipt_params, excluded_ref_vecs)

    def execute(
        self,
        data : StarlightSubtractionData,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        timestamps_sec : Optional[np.ndarray] = None,
        decomposition: Optional[learning.PrecomputedDecomposition] = None,
    ):
        image_vecs, probe_model_vecs = unwrap_inputs_to_matrices(data.inputs)
        med_vec = np.median(image_vecs, axis=0)
        image_vecs_medsub = image_vecs - med_vec
        params_kt = starlight_subtraction.KlipTParams(
            k_modes=k_modes,
            **self.construct_klipt_params_dict()
        )
        if probe_model_vecs is not None:
            probe_model_vecs_medsub = probe_model_vecs - med_vec
        else:
            probe_model_vecs_medsub = None
        image_vecs_resid, model_vecs_resid, decomposition = starlight_subtraction.klip_transpose(
            image_vecs_medsub, probe_model_vecs_medsub, decomposition,
            klipt_params=params_kt
        )
        return image_vecs_resid, model_vecs_resid, decomposition, med_vec, angles



@xconf.config
class KlipTransposePipeline(KlipTranspose, Pipeline):
    image_vecs : FitsConfig = xconf.field(help="2D array of observation vectors")
    model_vecs : FitsConfig = xconf.field(help="2D array of model signal vectors")
    basis_vecs : Optional[FitsConfig] = xconf.field(default=None, help="2D array of basis vectors")
    k_modes : int = xconf.field(default=5, help="Number of modes to subtract")

    def execute(self):
        basis_vecs = self.basis_vecs.load() if self.basis_vecs is not None else None
        basis = learning.PrecomputedDecomposition(None, None, mtx_v0=basis_vecs)
        super().execute(self.image_vecs.load(), self.model_vecs.load(), basis, self.k_modes)

def unwrap_inputs_to_matrices(klip_inputs: list[PipelineInput]) -> tuple[np.ndarray, Optional[list[np.ndarray]]]:
    matrices = []
    signal_matrices = []
    has_model = [ki.model_arr is not None for ki in klip_inputs]
    if any(has_model) and not all(has_model):
            raise ValueError("Some inputs have signal arrays, some don't")

    for idx, input_data in enumerate(klip_inputs):
        mtx_x = improc.unwrap_cube(
            input_data.sci_arr, input_data.estimation_mask
        )
        matrices.append(mtx_x)
        log.debug(
            f"klip input {idx} has {mtx_x.shape=} from {input_data.sci_arr.shape=} and "
            f"{np.count_nonzero(input_data.estimation_mask)=}"
        )
        if input_data.model_inputs is not None:
            mtx_x_signal_only = improc.unwrap_cube(
                input_data.model_arr,
                input_data.estimation_mask
            )
            signal_matrices.append(mtx_x_signal_only)

    mtx_x = np.vstack(matrices)
    signal_matrix = np.vstack(signal_matrices)
    return mtx_x, signal_matrix

def residuals_matrix_to_outputs(
    subtracted_mtx: np.ndarray, pipeline_inputs: list[PipelineInput], mean_vec: Optional[np.ndarray]=None,
    signal_mtx: Optional[np.ndarray]=None, fill_value=np.nan
) -> list[PipelineOutput]:
    start_idx = 0
    # cubes, signal_arrs, mean_images = [], [], []
    pipeline_outputs = []
    for input_data in pipeline_inputs:
        # first axis selects "which axis", second axis has an entry per retained pixel
        n_features = np.count_nonzero(input_data.estimation_mask)
        end_idx = start_idx + n_features
        # slice out the range of rows in the combined matrix that correspond to this input
        submatrix = subtracted_mtx[start_idx:end_idx]
        # log.debug(f"{submatrix=}")
        cube = improc.wrap_matrix(
            submatrix,
            input_data.estimation_mask,
            fill_value=fill_value,
        )
        if signal_mtx is not None:
            signal = improc.wrap_matrix(
                signal_mtx[start_idx:end_idx],
                input_data.estimation_mask,
                fill_value=fill_value,
            )
        else:
            signal = None

        if mean_vec is not None:
            sub_mean_vec = mean_vec[start_idx:end_idx]
            mean_image = improc.wrap_vector(
                sub_mean_vec,
                input_data.estimation_mask,
                fill_value=fill_value
            )
        else:
            mean_image = None
        pipeline_outputs.append(PipelineOutput(cube, input_data.destination_exts, signal, mean_image))
        start_idx += n_features
    return pipeline_outputs

@dataclass
class PostFilteringResult:
    image : np.ndarray
    kernel_diameter_px : Union[float,int]
    kernel : Optional[np.ndarray] = None

@dataclass
class PostFilteringResults:
    tophat : PostFilteringResult
    unfiltered_image : np.ndarray
    matched : PostFilteringResult
    coverage_count: np.ndarray

@dataclass
class StarlightSubtractModesResult:
    destination_images : dict[str, PostFilteringResults]
    pipeline_outputs : Optional[list[PipelineOutput]]
    pre_stack_filtered : Optional[list[PipelineOutput]]
    modes_requested : Union[int, float]

@xconf.config
class PrecomputedDecompositionConfig:
    mtx_u0 : Optional[FitsConfig] = xconf.field(default=None, help="2D array of left singular vectors")
    diag_s0 : Optional[FitsConfig] = xconf.field(default=None, help="1D array of singular values")
    mtx_v0 : Optional[FitsConfig] = xconf.field(default=None, help="2D array of right singular vectors")

    def load(self):
        return learning.PrecomputedDecomposition(
            mtx_u0=self.mtx_u0.load() if self.mtx_u0 is not None else None,
            diag_s0=self.diag_s0.load() if self.diag_s0 is not None else None,
            mtx_v0=self.mtx_v0.load() if self.mtx_v0 is not None else None,
        )

@dataclass
class StarlightSubtractResult:
    modes : dict[int, StarlightSubtractModesResult]
    pipeline_inputs : Optional[list[PipelineInput]]
    decomposition : Optional[learning.PrecomputedDecomposition]

@xconf.config
class CompanionSpiral:
    min_r_px : float = xconf.field(help="Input data to simultaneously reduce")
    max_r_px : float = xconf.field(help="Input data to simultaneously reduce")
    n_steps : int = xconf.field(help="Input data to simultaneously reduce")
    exclude_deg : float = xconf.field(help="")
    scale : Union[list[float],float] = xconf.field(help="")
    n_arms : int = xconf.field(default=1, help="")
    start_pa_deg : Union[list[float],float] = xconf.field(default=0, help="")

    def load(self):
        from xpipeline.tasks.characterization import CompanionSpec
        spiral = []
        delta_px = self.max_r_px - self.min_r_px
        if not isinstance(self.scale, list):
            scales = self.n_steps * [self.scale,]
        else:
            scales = self.scale
        for ring_idx in range(self.n_steps):
            r_px = self.min_r_px + (ring_idx / self.n_steps) * delta_px
            deg_per_arm = 360 / self.n_arms
            for arm_idx in range(self.n_arms):
                theta_deg = deg_per_arm * arm_idx + np.rad2deg(ring_idx * self.exclude_deg) + self.start_pa_deg
                spiral.append(CompanionSpec(r_px, theta_deg, scales[ring_idx]))
        return spiral

@xconf.config
class StarlightSubtractionDataConfig:
    inputs : list[PipelineInputConfig] = xconf.field(help="Input data to simultaneously reduce")
    angles : Union[FitsConfig,FitsTableColumnConfig,None] = xconf.field(help="1-D array or table column of derotation angles")
    times_sec : Union[FitsConfig,FitsTableColumnConfig,None] = xconf.field(default=None,help="1-D array or table column of observation times in seconds (to identify chunks)")
    coadd_chunk_size : int = xconf.field(default=1, help="Number of frames per coadded chunk (last chunk may be fewer)")
    coadd_operation : constants.CombineOperation = xconf.field(default=constants.CombineOperation.SUM, help="NaN-safe operation with which to combine coadd chunks")
    decimate_frames_by : int = xconf.field(default=1, help="Keep every Nth frame")
    decimate_frames_offset : int = xconf.field(default=0, help="Slice to begin decimation at this frame")
    companions : Union[CompanionSpiral,list[CompanionConfig]] = xconf.field(default_factory=lambda: [CompanionConfig(r_px=30, pa_deg=0, scale=0)], help="Companion amplitude and location to inject (scale 0 for no injection) and probe")

    def load(self) -> StarlightSubtractionData:
        angles = self.angles.load()[self.decimate_frames_offset::self.decimate_frames_by] if self.angles is not None else None
        times_sec = self.times_sec.load()[self.decimate_frames_offset::self.decimate_frames_by] if self.times_sec is not None else None
        if isinstance(self.companions, list):
            companions = [companion.to_companionspec() for companion in self.companions]
        else:
            companions = self.companions.load()
        model_gen_sec = 0
        pipeline_inputs = []
        for pinputconfig in self.inputs:
            if pinputconfig.model_inputs is None:
                raise ValueError(f"Pipeline input has no model information")
            pinput = pinputconfig.load()
            pinput.sci_arr = pinput.sci_arr[::self.decimate_frames_by].copy()
            if pinput.model_inputs.scale_factors is not None:
                pinput.sci_arr /= pinput.model_inputs.scale_factors[self.decimate_frames_offset::self.decimate_frames_by, np.newaxis, np.newaxis]
            ts = time.time()
            pinput.model_arr = np.zeros_like(pinput.sci_arr)
            for companion in companions:
                one_model_arr = generate_signal(
                    pinput.sci_arr.shape,
                    companion.r_px,
                    companion.pa_deg,
                    pinput.model_inputs.arr,
                    angles,
                )
                pinput.model_arr += one_model_arr
                if companion.scale != 0:
                    pinput.sci_arr += companion.scale * one_model_arr
            dt = time.time() - ts
            model_gen_sec += dt
            if self.coadd_chunk_size > 1:
                pinput.model_arr = improc.downsample_first_axis(pinput.model_arr, self.coadd_chunk_size, self.coadd_operation)
                pinput.sci_arr = improc.downsample_first_axis(pinput.sci_arr, self.coadd_chunk_size, self.coadd_operation)
            pipeline_inputs.append(pinput)
        log.debug("Spent %f seconds in model generation", model_gen_sec)

        if self.coadd_chunk_size > 1:
            angles = improc.downsample_first_axis(angles, self.coadd_chunk_size, constants.CombineOperation.MEAN)

        return StarlightSubtractionData(
            inputs=pipeline_inputs,
            angles=angles,
            companions=companions,
            times_sec=times_sec,
        )

@xconf.config
class KModesValuesConfig:
    values: list[int] = xconf.field(help="Which values to try for number of modes to subtract")

    def as_request_value_pairs(self, max_rank: int):
        '''Returns (value, value) pairs for each entry in `values` that's < max_rank'''
        values = [(x, x) for x in self.values if x < max_rank]
        if not len(values):
            raise ValueError(f"Given {max_rank=}, no valid values from {self.values}")
        return values


@xconf.config
class KModesRangeConfig:
    start: int = xconf.field(help="Which values to try for number of modes to subtract")
    stop: int = xconf.field(help="Which values to try for number of modes to subtract")
    step: int = xconf.field(help="Which values to try for number of modes to subtract")

    def as_request_value_pairs(self, max_rank: int):
        '''Returns (value, value) pairs for each entry in `values` that's < max_rank'''
        values = [(x, x) for x in range(self.start, self.stop, self.step) if x < max_rank]
        if not len(values):
            raise ValueError(f"Given {max_rank=}, no valid values from {self.values}")
        return values


@xconf.config
class KModesFractionConfig:
    fractions: list[float] = xconf.field(default_factory=lambda: [0.1], help="Fraction of the maximum number of modes to subtract in (0, 1.0)")

    def as_request_value_pairs(self, max_rank: int):
        '''Returns (requested fraction, actual value) pairs for each entry in `fractions`'''
        if any(x >= 1.0 for x in self.fractions):
            raise ValueError(f"Invalid fractions in config: {self.fractions} (must be 0 < x < 1)")
        values = [(x, int(x * max_rank)) for x in self.fractions]
        return values

KModesConfig = Union[KModesValuesConfig,KModesFractionConfig,KModesRangeConfig]

@xconf.config
class PreStackFilter:
    def execute(
        self,
        pipeline_outputs: list[PipelineOutput],
        measurement_location: CompanionSpec,
        resolution_element_px : float,
        derotation_angles: Optional[np.ndarray]=None,
    ) -> list[PipelineOutput]:
        return pipeline_outputs


@xconf.config
class NoOpPreStackFilter(PreStackFilter):
    no_op : bool = xconf.field(help="Do not pre-filter before stacking")

@xconf.config
class TophatPreStackFilter:
    tophat : bool = xconf.field(default=True, help="")
    def execute(
        self,
        pipeline_outputs: list[PipelineOutput],
        measurement_location: CompanionSpec,
        resolution_element_px : float,
        derotation_angles: Optional[np.ndarray]=None,
    ) -> list[PipelineOutput]:
        out = []
        kernel = Tophat2DKernel(radius=resolution_element_px)
        for po in pipeline_outputs:
            sci_arr_filtered = np.zeros_like(po.sci_arr)
            for i in range(po.sci_arr.shape[0]):
                sci_arr_filtered[i] = convolve_fft(
                    po.sci_arr[i],
                    kernel,
                    nan_treatment='fill',
                    fill_value=0.0,
                    preserve_nan=True,
                )
            out.append(PipelineOutput(sci_arr_filtered, po.destination_exts, model_arr=po.model_arr))
        return out

@xconf.config
class MatchedPreStackFilter(PreStackFilter):
    kernel_diameter_resel : float = xconf.field(default=1.5, help="Diameter in resolution elements beyond which matched filter kernel is set to zero to avoid spurious detections")
    matched : bool = xconf.field(default=True, help="")
    def execute(
        self,
        pipeline_outputs: list[PipelineOutput],
        measurement_location: CompanionSpec,
        resolution_element_px : float,
        derotation_angles: Optional[np.ndarray]=None,
    ) -> list[PipelineOutput]:
        out = []
        for po in pipeline_outputs:
            rho, _ = improc.polar_coords(improc.arr_center(po.sci_arr.shape[1:]), po.sci_arr.shape[1:])
            sci_arr_filtered = np.zeros_like(po.sci_arr)
            for i in range(po.sci_arr.shape[0]):
                angle = derotation_angles[i] if derotation_angles is not None else 0
                # shift kernel to center
                dx, dy = characterization.r_pa_to_x_y(measurement_location.r_px, measurement_location.pa_deg, derotation_angle_deg=angle)
                kernel = improc.shift2(po.model_arr[i], -dx, -dy)

                # trim kernel to only central region
                radius_px = (self.kernel_diameter_resel * resolution_element_px) / 2
                kernel[rho > radius_px] = 0

                # normalize kernel and flip to form matched filter
                kernel /= np.nansum(kernel**2)
                kernel = np.flip(kernel, axis=(0, 1))
                sci_arr_filtered[i] = convolve_fft(
                    po.sci_arr[i],
                    kernel,
                    normalize_kernel=False,
                    nan_treatment='fill',
                    fill_value=0.0,
                    preserve_nan=True,
                )
            out.append(PipelineOutput(sci_arr_filtered, po.destination_exts, model_arr=po.model_arr))
        return out

@dataclass
class ImageStackingResult:
    image : np.ndarray
    coverage : np.ndarray

@xconf.config
class ImageStack:
    combine : constants.CombineOperation = xconf.field(default=constants.CombineOperation.MEDIAN, help="How to combine image for stacking")
    minimum_coverage: int = xconf.field(default=1, help="Number of overlapping source frames covering a derotated frame pixel for it to be kept in the final image")


    def execute(self, pipeline_outputs: list[PipelineOutput], angles: Optional[np.ndarray]) -> tuple[dict[str, ImageStackingResult], dict[str, list[PipelineOutput]]]:
        # group outputs by their destination extension
        outputs_by_ext = defaultdict(list)
        for po in pipeline_outputs:
            for destination_ext in po.destination_exts:
                outputs_by_ext[destination_ext].append(po)
        destination_images = {}
        for ext in outputs_by_ext:
            log.debug(f"Stacking {len(outputs_by_ext[ext])} outputs for {ext=}")
            # construct a single outputs cube with all contributing inputs
            all_outputs_cube = None
            for po in outputs_by_ext[ext]:
                mask_good = ~(np.nanmin(po.sci_arr, axis=(1,2)) == np.nanmax(po.sci_arr, axis=(1,2)))
                good_frames = np.count_nonzero(mask_good)
                log.debug(f"This pipeline output for {ext} has {good_frames=}")
                if good_frames == 0:
                    continue
                assert isinstance(po, PipelineOutput)
                derot_cube = improc.derotate_cube(po.sci_arr[mask_good], angles[mask_good])
                if all_outputs_cube is None:
                    all_outputs_cube = derot_cube
                else:
                    all_outputs_cube = np.concatenate([all_outputs_cube, derot_cube])
            # combine the concatenated cube into a single plane
            if all_outputs_cube is None:
                log.info(f"Processing destination extension {ext}: After excluding frames without usable pixels, no output remains to stack")
                all_outputs_cube = np.nan * np.ones((1,) + po.sci_arr.shape[1:])
            finim = improc.combine(all_outputs_cube, self.combine)
            # apply minimum coverage mask
            finite_elements_cube = np.isfinite(all_outputs_cube)
            coverage_count = np.sum(finite_elements_cube, axis=0)
            coverage_mask = coverage_count > self.minimum_coverage
            finim[~coverage_mask] = np.nan
            destination_images[ext] = ImageStackingResult(image=finim, coverage=coverage_count)
        return destination_images, outputs_by_ext


@xconf.config
class  _BasePostFilter:
    kernel_diameter_px : float = xconf.field(default=None, help="Filter kernel radius for spacing signal estimation apertures")
    def execute(
        self,
        destination_image: np.ndarray,
        pipeline_outputs_for_ext: list[PipelineOutput],
        measurement_location : CompanionSpec,
        resolution_element_px : float,
        derotation_angles: Optional[np.ndarray] = None,
    ) -> PostFilteringResult:
        raise NotImplementedError("Subclasses must implement execute()")

@xconf.config
class MatchedPostFilter(_BasePostFilter):
    kernel_diameter_resel : float = xconf.field(default=1.5, help="Diameter in resolution elements beyond which matched filter kernel is set to zero to avoid spurious detections")
    combine : constants.CombineOperation = xconf.field(default=constants.CombineOperation.MEAN, help="How to combine model residuals for stacking")

    def execute(
        self,
        destination_image: np.ndarray,
        pipeline_outputs_for_ext: list[PipelineOutput],
        measurement_location : CompanionSpec,
        resolution_element_px : float,
        derotation_angles: Optional[np.ndarray] = None,
    ) -> PostFilteringResult:
        derotated_cube = None
        if derotation_angles is None:
            raise NotImplementedError('need to handle no-derot case')
        sci_arr = pipeline_outputs_for_ext[0].sci_arr
        nobs = sci_arr.shape[0]
        derotated_cube = np.zeros(
            (len(pipeline_outputs_for_ext) * nobs,) + sci_arr.shape[1:],
            dtype=sci_arr.dtype
        )
        for idx, output in enumerate(pipeline_outputs_for_ext):
            log.debug(f"Derotating model residuals from output {idx + 1} / {len(pipeline_outputs_for_ext)}")
            improc.derotate_cube(output.model_arr, derotation_angles, output=derotated_cube[idx * nobs:(idx+1) * nobs])
        kernel = improc.combine(derotated_cube, self.combine)
        del derotated_cube

        # shift kernel to center
        dx, dy = characterization.r_pa_to_x_y(measurement_location.r_px, measurement_location.pa_deg, 0, 0)
        kernel = improc.shift2(kernel, -dx, -dy)

        # trim kernel to only central region
        radius_px = (self.kernel_diameter_resel * resolution_element_px) / 2
        log.debug(f"Matched filter with {self.kernel_diameter_resel} lambda/D extent is {radius_px=} given {resolution_element_px=}")
        rho, _ = improc.polar_coords(improc.arr_center(kernel), kernel.shape)
        kernel[rho > radius_px] = 0

        if np.any(~np.isfinite(kernel)):
            return PostFilteringResult(
                kernel=kernel,
                image=np.nan * destination_image,
                kernel_diameter_px=2 * radius_px,
            )

        # normalize kernel and flip to form matched filter
        kernel /= np.nansum(kernel**2)
        kernel = np.flip(kernel, axis=(0, 1))

        # apply
        filtered_image = convolve_fft(
            destination_image,
            kernel,
            normalize_kernel=False,
            nan_treatment='fill',
            fill_value=0.0,
            preserve_nan=True,
        )
        return PostFilteringResult(
            kernel=kernel,
            image=filtered_image,
            kernel_diameter_px=2 * radius_px,
        )

@xconf.config
class TophatPostFilter(_BasePostFilter):
    kernel_radius_px : float = xconf.field(default=None, help="Filter kernel radius for spacing signal estimation apertures, default is use resolution_element_px value passed in")
    tophat_filter : bool = xconf.field(default=True, help="Supply 'tophat_filter' to explicitly select TophatPostFilter")
    exclude_nearest_apertures : int = xconf.field(default=1, help="Exclude this many apertures on either side of the measurement location from the noise sample")

    def execute(
        self,
        destination_image: np.ndarray,
        pipeline_outputs_for_ext: list[PipelineOutput],
        measurement_location : CompanionSpec,
        resolution_element_px : float,
        derotation_angles: Optional[np.ndarray] = None,
    ) -> PostFilteringResult:
        radius_px = resolution_element_px / 2 if self.kernel_radius_px is None else self.kernel_radius_px
        kernel = Tophat2DKernel(radius=radius_px)
        filtered_image = convolve_fft(
            destination_image,
            kernel,
            nan_treatment='fill',
            fill_value=0.0,
            preserve_nan=True,
        )
        return PostFilteringResult(
            kernel=kernel.array,
            image=filtered_image,
            kernel_diameter_px=radius_px * 2,
        )

def signal_from_filter(img, mf):
    mf /= np.nansum(mf**2)
    mf = np.flip(mf, axis=(0, 1))
    mf[np.abs(mf) < 0.3 * np.nanmax(mf)] = 0
    # convolve kernel
    filtered_image = convolve_fft(
        img,
        mf,
        normalize_kernel=False,
    )
    # measure signal at center, since we're not translating the matched filter kernel to the companion location
    yctr, xctr = math.ceil((filtered_image.shape[0] - 1) / 2), math.ceil((filtered_image.shape[1] - 1) / 2)
    return filtered_image[int(yctr), int(xctr)]

@dataclass
class StarlightSubtractionMeasurement:
    signal : float
    snr : float
    r_px : float
    pa_deg : float

@dataclass
class StarlightSubtractionFilterMeasurement:
    locations : list[StarlightSubtractionMeasurement]
    post_filtering_result : Optional[PostFilteringResult]

@dataclass
class StarlightSubtractionFilterMeasurements:
    tophat : StarlightSubtractionFilterMeasurement
    matched : StarlightSubtractionFilterMeasurement
    coverage_count : np.ndarray
    unfiltered_image : np.ndarray

@xconf.config
class PostFilter:
    tophat : TophatPostFilter = xconf.field(default=TophatPostFilter(), help="Filter final derotated images with a circular aperture")
    matched : MatchedPostFilter = xconf.field(default=MatchedPostFilter(), help="Filter final derotated images with a matched filter based on the model PSF")

    def execute(
        self,
        destination_image: np.ndarray,
        coverage_count: np.ndarray,
        pipeline_outputs_for_ext: list[PipelineOutput],
        measurement_location : CompanionSpec,
        resolution_element_px : float,
        derotation_angles: Optional[np.ndarray] = None,
    ):
        tophat_res = self.tophat.execute(destination_image, pipeline_outputs_for_ext, measurement_location, resolution_element_px, derotation_angles)
        matched_res = self.matched.execute(destination_image, pipeline_outputs_for_ext, measurement_location, resolution_element_px, derotation_angles)
        return PostFilteringResults(
            tophat=tophat_res,
            matched=matched_res,
            unfiltered_image=destination_image,
            coverage_count=coverage_count,
        )

@xconf.config
class StarlightSubtract:
    strategy : Union[KlipTranspose,Klip,KlipSubspace,DynamicModeDecomposition] = xconf.field(help="Strategy with which to estimate and subtract starlight")
    resolution_element_px : float = xconf.field(help="One resolution element (lambda / D) in pixels")
    image_stack: ImageStack = xconf.field(default=ImageStack(), help="How to combine images after starlight subtraction and filtering")
    pre_stack_filter : Union[MatchedPreStackFilter,TophatPreStackFilter,NoOpPreStackFilter] = xconf.field(default=None, help="Process after removing starlight and before stacking")
    k_modes : KModesConfig = xconf.field(default_factory=KModesFractionConfig, help="Which values to try for number of modes to subtract")
    post_filter : PostFilter = xconf.field(default=PostFilter())
    return_inputs : bool = xconf.field(default=False, help="Whether original images before starlight subtraction should be returned")
    return_pre_stack_filtered : bool = xconf.field(default=False, help="Whether filtered images before stacking should be returned")
    return_decomposition : bool = xconf.field(default=False, help="Whether the computed decomposition should be returned")
    return_residuals : bool = xconf.field(default=False, help="Whether residual images after starlight subtraction should be returned")

    def execute(self, data : StarlightSubtractionData) -> StarlightSubtractResult:
        destination_exts = defaultdict(list)
        for pinput in data.inputs:
            for destination_ext in pinput.destination_exts:
                if len(destination_exts[destination_ext]):
                    if pinput.sci_arr.shape != destination_exts[destination_ext][0].sci_arr.shape:
                        raise ValueError(f"Dimensions of current science array {pinput.sci_arr.shape=} mismatched with others. Use separate destination_ext settings for each input, or make them the same size.")
                destination_exts[destination_ext].append(pinput)
        max_rank = data.max_rank()
        k_modes_requested_and_values = self.k_modes.as_request_value_pairs(max_rank)
        log.debug(f"Estimation masks and data cubes imply max rank {max_rank}, implying {k_modes_requested_and_values=} from {self.k_modes}")

        max_k_modes = max(val for _, val in k_modes_requested_and_values)
        log.debug(f"Computing basis with {max_k_modes=}")
        decomp = self.strategy.prepare(
            data,
            max_k_modes,
            angles=data.angles,
        )

        results_for_modes = {}
        for mode_idx, (k_modes_requested, k_modes) in enumerate(k_modes_requested_and_values):
            log.debug(f"Subtracting starlight for modes value k={k_modes} ({mode_idx+1}/{len(k_modes_requested_and_values)})")
            res = self.strategy.execute(
                data,
                k_modes,
                angles=data.angles,
                decomposition=decomp,
            )
            data_vecs_resid, model_vecs_resid, _, _, angles = res

            pipeline_outputs = residuals_matrix_to_outputs(
                data_vecs_resid,
                data.inputs,
                mean_vec=None,
                signal_mtx=model_vecs_resid,
            )
            # filter individual frames before stacking
            if self.pre_stack_filter is not None:
                filtered_outputs = self.pre_stack_filter.execute(
                    pipeline_outputs,
                    data.companions[0],
                    self.resolution_element_px,
                    angles
                )
            else:
                filtered_outputs = pipeline_outputs
            destination_stacking_results, outputs_by_ext = self.image_stack.execute(filtered_outputs, angles)
            # results_for_modes[k_modes] = res
            post_filtered_image_results = {}
            for dest_ext in destination_stacking_results:
                pfres : PostFilteringResults = self.post_filter.execute(
                    destination_stacking_results[dest_ext].image,
                    destination_stacking_results[dest_ext].coverage,
                    outputs_by_ext[dest_ext],
                    data.companions[0],
                    self.resolution_element_px,
                    angles
                )
                post_filtered_image_results[dest_ext] = pfres
            res = StarlightSubtractModesResult(
                destination_images=post_filtered_image_results,
                pre_stack_filtered=filtered_outputs if (self.pre_stack_filter and self.return_pre_stack_filtered) else None,
                pipeline_outputs=pipeline_outputs if self.return_residuals else None,
                modes_requested=k_modes_requested,
            )
            results_for_modes[k_modes_requested] = res


        return StarlightSubtractResult(
            modes=results_for_modes,
            pipeline_inputs=data.inputs if self.return_inputs else None,
            decomposition=decomp if self.return_decomposition else None,
        )

@xconf.config
class StarlightSubtractPipeline(StarlightSubtract, Pipeline):
    data : StarlightSubtractionDataConfig = xconf.field(help="Starlight subtraction data")
    def execute(self) -> StarlightSubtractResult:
        data = self.data.load()
        return super().execute(data)



@dataclass
class StarlightSubtractionMeasurementSet:
    by_ext : dict[str, StarlightSubtractionFilterMeasurements]

@dataclass
class StarlightSubtractionMeasurements:
    companions : list[CompanionSpec]
    by_modes: dict[int,StarlightSubtractionMeasurementSet]
    subtraction_result : Optional[StarlightSubtractResult]
    modes_chosen_to_requested_lookup : dict[int, Union[int, float]]

@xconf.config
class MeasureStarlightSubtraction:
    # resolution_element_px : float = xconf.field(help="Diameter of the resolution element in pixels")
    exclude_nearest_apertures: int = xconf.field(default=1, help="How many locations on either side of the probed companion location should be excluded from the SNR calculation")
    subtraction : StarlightSubtract = xconf.field(help="Configure starlight subtraction options")
    return_starlight_subtraction : bool = xconf.field(default=False, help="whether to return the starlight subtraction result")
    return_post_filtering_result : bool = xconf.field(default=False, help="whether to return the post-filtered image")
    frames_per_chunk : int = xconf.field(default=0, help="Construct chunks of N frames to reduce independently and combine with averaging at the end, 0 to disable")
    chunk_stride : int = xconf.field(default=0, help="Offset between successive chunks, 0 for 'same as chunk size'. Permits overlapping chunks.")

    def _combine_starlight_subtract_modes_results(self, intermediate_ssresults : list[StarlightSubtractResult]):
        k_modes_requested_values = list(sorted(intermediate_ssresults[0].modes.keys()))
        destination_exts = list(sorted(intermediate_ssresults[0].modes[k_modes_requested_values[0]].destination_images.keys()))
        combined_results_by_modes = {}
        for k_modes_requested in k_modes_requested_values:
            destination_images = {}
            for ext in destination_exts:
                unfiltered_images = []
                filter_stacks : dict[str, list[PostFilteringResult]] = {}

                coverage_overall = None
                for ssresult in intermediate_ssresults:
                    pfresults : PostFilteringResults = ssresult.modes[k_modes_requested].destination_images[ext]
                    unfiltered_images.append(pfresults.unfiltered_image)
                    if coverage_overall is None:
                        coverage_overall = pfresults.coverage_count.copy()
                    else:
                        coverage_overall += pfresults.coverage_count
                    for filter_name in POST_FILTER_NAMES:
                        filter_result : PostFilteringResult = getattr(pfresults, filter_name)
                        if filter_name not in filter_stacks:
                            filter_stacks[filter_name] = [filter_result]
                        else:
                            filter_stacks[filter_name].append(filter_result)

                pfr_by_filter_name = {}
                for filter_name in filter_stacks:
                    pfresults : list[PostFilteringResult] = filter_stacks[filter_name]
                    stacked_images = np.stack([pfr.image for pfr in pfresults])
                    kernels = np.stack([pfr.kernel for pfr in pfresults])
                    reduced_image = np.nanmean(stacked_images, axis=0)
                    pfr_by_filter_name[filter_name] = PostFilteringResult(
                        reduced_image,
                        kernel_diameter_px=pfresults[0].kernel_diameter_px,
                        kernel=kernels,
                    )
                destination_images[ext] = PostFilteringResults(
                    unfiltered_image=np.nansum(unfiltered_images, axis=0),
                    coverage_count=coverage_overall,
                    **pfr_by_filter_name,
                )

            combined_results_by_modes[k_modes_requested] = StarlightSubtractModesResult(
                destination_images=destination_images,
                pipeline_outputs=None,
                pre_stack_filtered=None,
                modes_requested=k_modes_requested,
            )

        ssresult : StarlightSubtractResult = StarlightSubtractResult(
            modes=combined_results_by_modes,
            pipeline_inputs=None,
            decomposition=None,
        )

        return ssresult

    def execute(
        self, data : StarlightSubtractionData
    ) -> StarlightSubtractionMeasurements:
        if self.frames_per_chunk > 0:
            if self.return_starlight_subtraction:
                raise NotImplementedError("TODO")

            chunk_stride = self.frames_per_chunk if self.chunk_stride == 0 else self.chunk_stride
            slices = utils.compute_chunk_slices(
                data.observations_count(),
                self.frames_per_chunk,
                chunk_stride
            )
            log.debug(f"Got {len(slices)} slices in chunk mode: {slices}")
        else:
            slices = [slice(None)]
        if len(slices) > 1:
            intermediate_ssresults = []
            for the_slice in slices:
                log.debug(f"Using chunked mode, processing {the_slice}")
                subset_data : StarlightSubtractionData = data.from_slice(the_slice)
                intermediate_ssresults.append(self.subtraction.execute(subset_data))
            ssresult : StarlightSubtractResult = self._combine_starlight_subtract_modes_results(intermediate_ssresults)
        else:
            ssresult : StarlightSubtractResult = self.subtraction.execute(data)
        result = StarlightSubtractionMeasurements(
            companions=data.companions,
            by_modes={},
            subtraction_result=ssresult if self.return_starlight_subtraction else None,
            modes_chosen_to_requested_lookup={}
        )
        for k_modes in ssresult.modes:
            meas = StarlightSubtractionMeasurementSet(by_ext={})
            for ext in ssresult.modes[k_modes].destination_images:
                pfresults : PostFilteringResults = ssresult.modes[k_modes].destination_images[ext]
                coverage_mask = pfresults.coverage_count > self.subtraction.image_stack.minimum_coverage
                log.debug(f"Measuring SNR from {np.count_nonzero(coverage_mask)} pixels with coverage > {self.subtraction.image_stack.minimum_coverage}")
                measurements = {}
                for filter_name in POST_FILTER_NAMES:
                    companion_measurements = []
                    filter_result : PostFilteringResult = getattr(pfresults, filter_name)
                    for companion in data.companions:
                        snr, signal = snr_from_convolution(
                            filter_result.image,
                            loc_rho=companion.r_px,
                            loc_pa_deg=companion.pa_deg,
                            aperture_diameter_px=filter_result.kernel_diameter_px,
                            exclude_nearest=self.exclude_nearest_apertures,
                            good_pixel_mask=coverage_mask,
                        )
                        cm = StarlightSubtractionMeasurement(
                            signal=signal,
                            snr=snr,
                            r_px=companion.r_px,
                            pa_deg=companion.pa_deg,
                        )
                        companion_measurements.append(cm)
                    measurements[filter_name] = StarlightSubtractionFilterMeasurement(
                        locations=companion_measurements,
                        post_filtering_result=filter_result if self.return_post_filtering_result else None,
                    )
                measurements = StarlightSubtractionFilterMeasurements(
                    unfiltered_image=pfresults.unfiltered_image,
                    coverage_count=pfresults.coverage_count,
                    **measurements
                )
                meas.by_ext[ext] = measurements
            result.by_modes[k_modes] = meas
            result.modes_chosen_to_requested_lookup[k_modes] = ssresult.modes[k_modes].modes_requested
        return result

    def measurements_to_jsonable(self, res : StarlightSubtractionMeasurements, k_modes_values):
        output_dict = {}
        output_dict['config'] = xconf.asdict(self)
        output_dict['results'] = {
            'k_modes_values': k_modes_values,
            'companions': [],
        }
        for spec in res.companions:
            output_dict['results']['companions'].append({
                'r_px': spec.r_px,
                'pa_deg': spec.pa_deg,
                'scale': spec.scale,
            })

        for k in k_modes_values:
            for ext in res.by_modes[k].by_ext:
                if ext not in output_dict['results']:
                    output_dict['results'][ext] = []
                filtered_measurements = {}
                for filter_name in POST_FILTER_NAMES:
                    filter_res : StarlightSubtractionFilterMeasurement = getattr(res.by_modes[k].by_ext[ext], filter_name)
                    loc_measurements = []
                    for loc in filter_res.locations:
                        loc_measurements.append({
                            'snr': loc.snr,
                            'signal': loc.signal,
                        })
                    filtered_measurements[filter_name] = loc_measurements
                output_dict['results'][ext].append(filtered_measurements)
        return output_dict

@xconf.config
class MeasureStarlightSubtractionPipeline(MeasureStarlightSubtraction):
    data : StarlightSubtractionDataConfig = xconf.field(help="Starlight subtraction data")
    def execute(self) -> StarlightSubtractionMeasurements:
        data = self.data.load()
        return super().execute(data)

@xconf.config
class SaveMeasuredStarlightSubtraction:
    save_decomposition : bool = xconf.field(default=False, help="Whether to save data decomposition components as a FITS file")
    save_residuals : bool = xconf.field(default=False, help="Whether to save starlight subtraction residuals as a FITS file")
    save_inputs : bool = xconf.field(default=False, help="Whether to save input data (post-injection) and model as a FITS file")
    save_unfiltered_images : bool = xconf.field(default=False, help="Whether to save stacked but unfiltered images")
    save_pre_stack_filtered_images : bool = xconf.field(default=False, help="Whether to save image sequences that have been filtered before being stacked")
    save_post_filtering_images : bool = xconf.field(default=False, help="Whether to save stacked and post-filtering images")
    save_filter_kernels : bool = xconf.field(default=False, help="Save matched filter kernels")
    save_coverage_map : bool = xconf.field(default=False, help="Whether to save a coverage map counting the frames contributing to each pixel")
    save_ds9_regions : bool = xconf.field(default=False, help="Whether to write a ds9 region file for the signal estimation pixels")

    def execute(self, data : StarlightSubtractionData, measure_subtraction: MeasureStarlightSubtraction, destination : DirectoryConfig) -> StarlightSubtractionMeasurements:
        if self.save_decomposition:
            measure_subtraction.return_starlight_subtraction = True
            measure_subtraction.subtraction.return_decomposition = True
        if self.save_residuals:
            measure_subtraction.return_starlight_subtraction = True
            measure_subtraction.subtraction.return_residuals = True
        if self.save_inputs:
            measure_subtraction.return_starlight_subtraction = True
            measure_subtraction.subtraction.return_inputs = True
        if self.save_unfiltered_images:
            measure_subtraction.return_starlight_subtraction = True
        if self.save_post_filtering_images or self.save_filter_kernels:
            measure_subtraction.return_post_filtering_result = True
        if self.save_ds9_regions:
            measure_subtraction.return_post_filtering_result = True
        if self.save_pre_stack_filtered_images:
            if measure_subtraction.subtraction.pre_stack_filter is None:
                raise RuntimeError("Must specify pre stack filter if saving images")
            measure_subtraction.subtraction.return_pre_stack_filtered = True

        output_filenames = {
            'decomposition.fits': self.save_decomposition,
            'residuals.fits': self.save_residuals,
            'inputs.fits': self.save_inputs,
            'unfiltered.fits': self.save_unfiltered_images,
            'post_filtering.fits': self.save_post_filtering_images,
            'filter_kernels.fits': self.save_filter_kernels,
            'pre_filtering.fits': self.save_pre_stack_filtered_images,
            'coverage.fits': self.save_coverage_map,
        }
        destination.ensure_exists()
        for fn, condition in output_filenames.items():
            if condition and destination.exists(fn):
                log.error(f"Output filename {fn} exists at {destination.join(fn)}")

        res : StarlightSubtractionMeasurements = measure_subtraction.execute(data)
        k_modes_values = list(res.by_modes.keys())
        n_inputs = len(data.inputs)
        output_dict = measure_subtraction.measurements_to_jsonable(res, k_modes_values)
        log.debug(pformat(output_dict))
        output_json = orjson.dumps(
            output_dict,
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2,
        )
        with destination.open_path('result.json', 'wb') as fh:
            fh.write(output_json)

        images_by_ext = {}
        if self.save_post_filtering_images or self.save_unfiltered_images:
            for k_modes in k_modes_values:
                for ext, postfilteringresults in res.by_modes[k_modes].by_ext.items():
                    if ext not in images_by_ext:
                        images_by_ext[ext] = []
                    images_by_ext[ext].append(postfilteringresults.unfiltered_image)
        if self.save_unfiltered_images:
            unfilt_hdul = fits.HDUList([
                fits.PrimaryHDU(),
            ])
            unfilt_hdul[0].header['K_MODES'] = ','.join(map(str, k_modes_values))
            for ext in images_by_ext:
                unfilt_hdul.append(fits.ImageHDU(
                    np.stack(images_by_ext[ext]),
                    name=ext
                ))
            with destination.open_path('unfiltered.fits', 'wb') as fh:
                unfilt_hdul.writeto(fh)

        if self.save_pre_stack_filtered_images:
            sci_arrays_by_output = [[] for i in range(n_inputs)]
            for k_modes in k_modes_values:
                for i in range(n_inputs):
                    sci_arrays_by_output[i].append(
                        res.subtraction_result.modes[k_modes].pre_stack_filtered[i].sci_arr
                    )
            prefilter_hdul = fits.HDUList([
                fits.PrimaryHDU(),
            ])
            prefilter_hdul[0].header['K_MODES'] = ','.join(map(str, k_modes_values))
            for i in range(n_inputs):
                ext = f"PRE_STACK_FILTERED_{i:02}"
                sci_array_stack = np.stack(sci_arrays_by_output[i])
                prefilter_hdul.append(fits.ImageHDU(sci_array_stack, name=ext))
            with destination.open_path('pre_stack_filtered.fits', 'wb') as fh:
                prefilter_hdul.writeto(fh)

        if self.save_coverage_map:
            coverage_hdul = fits.HDUList([
                fits.PrimaryHDU(),
            ])
            # coverage will be the same for all modes values, so pick
            # the first one
            for ext in res.by_modes[k_modes_values[0]].by_ext:
                res_for_ext = res.by_modes[k_modes_values[0]].by_ext[ext]
                coverage_count = res_for_ext.coverage_count
                coverage_hdul.append(fits.ImageHDU(coverage_count, name=f"COVERAGE_{ext}"))
            with destination.open_path(f"coverage.fits", "wb") as fh:
                coverage_hdul.writeto(fh)

        if self.save_post_filtering_images or self.save_ds9_regions or self.save_filter_kernels:
            images_by_filt_by_ext = {}
            kernels_by_filt_by_ext = {}
            for k_modes in k_modes_values:
                for ext, ss_result_by_filter in res.by_modes[k_modes].by_ext.items():
                    for filt_name in POST_FILTER_NAMES:
                        ss_result : StarlightSubtractionMeasurement = getattr(ss_result_by_filter, filt_name)
                        region_file_name = None
                        if filt_name not in images_by_filt_by_ext:
                            images_by_filt_by_ext[filt_name] = {}
                            kernels_by_filt_by_ext[filt_name] = {}
                        if ext not in images_by_filt_by_ext[filt_name]:
                            images_by_filt_by_ext[filt_name][ext] = []
                            kernels_by_filt_by_ext[filt_name][ext] = []
                        images_by_filt_by_ext[filt_name][ext].append(ss_result.post_filtering_result.image)
                        kernels_by_filt_by_ext[filt_name][ext].append(ss_result.post_filtering_result.kernel)
                        if self.save_ds9_regions and region_file_name is None:
                            from ..tasks import characterization, improc
                            yc, xc = improc.arr_center(images_by_filt_by_ext[filt_name][ext][0])
                            # ds9 is 1-indexed
                            yc += 1
                            xc += 1
                            region_specs = ""
                            kernel_diameter_px = ss_result.post_filtering_result.kernel_diameter_px
                            for companion in data.companions:
                                for idx, (x, y) in enumerate(characterization.simple_aperture_locations(
                                    companion.r_px,
                                    companion.pa_deg,
                                    kernel_diameter_px,
                                    measure_subtraction.exclude_nearest_apertures,
                                    xcenter=xc, ycenter=yc
                                )):
                                    region_specs += f"circle({x},{y},{kernel_diameter_px / 2 if filt_name != 'none' else 0.5}) # color={'red' if idx == 0 else 'green'}\n"
                            region_file_name = f"regions_{ext}_{filt_name}.reg"
                            with destination.open_path(region_file_name, "wb") as fh:
                                fh.write(region_specs.encode('utf8'))


            if self.save_post_filtering_images:
                postfilt_hdul = fits.HDUList([
                    fits.PrimaryHDU(),
                ])
                postfilt_hdul[0].header['K_MODES'] = ','.join(map(str, k_modes_values))
                for ext in images_by_ext:
                    postfilt_hdul.append(fits.ImageHDU(
                        images_by_ext[ext],
                        name=f"{ext}",
                    ))
                    for filt_name in images_by_filt_by_ext:
                        postfilt_hdul.append(fits.ImageHDU(
                            np.stack(images_by_filt_by_ext[filt_name][ext]),
                            name=f"{ext}_{filt_name}",
                        ))
                with destination.open_path(f'post_filtering.fits', 'wb') as fh:
                    postfilt_hdul.writeto(fh)

            if self.save_filter_kernels:
                kernels_hdul = fits.HDUList([
                    fits.PrimaryHDU(),
                ])
                kernels_hdul[0].header['K_MODES'] = ','.join(map(str, k_modes_values))
                for filt_name in images_by_filt_by_ext:
                    for ext in images_by_filt_by_ext[filt_name]:
                        kernels_hdul.append(fits.ImageHDU(
                            np.stack(kernels_by_filt_by_ext[filt_name][ext]),
                            name=f"{ext}_{filt_name}_KERNEL",
                        ))
                with destination.open_path(f'filter_kernels.fits', 'wb') as fh:
                    kernels_hdul.writeto(fh)



        if measure_subtraction.return_starlight_subtraction:
            if self.save_decomposition:
                decomp = res.subtraction_result.decomposition
                decomp_hdul = fits.HDUList([fits.PrimaryHDU(),])
                decomp_hdul.append(fits.ImageHDU(decomp.mtx_u0, name='MTX_U0'))
                decomp_hdul.append(fits.ImageHDU(decomp.diag_s0, name='DIAG_S0'))
                decomp_hdul.append(fits.ImageHDU(decomp.mtx_v0, name='MTX_V0'))
                with destination.open_path('decomposition.fits', 'wb') as fh:
                    decomp_hdul.writeto(fh)
            if self.save_residuals:
                sci_arrays_by_output = [[] for i in range(n_inputs)]
                model_arrays_by_output = [[] for i in range(n_inputs)]
                for k_modes in k_modes_values:
                    for i in range(n_inputs):
                        sci_arrays_by_output[i].append(
                            res.subtraction_result.modes[k_modes].pipeline_outputs[i].sci_arr
                        )
                        model_arrays_by_output[i].append(
                            res.subtraction_result.modes[k_modes].pipeline_outputs[i].model_arr
                        )
                if self.save_residuals:
                    resid_hdul = fits.HDUList([
                        fits.PrimaryHDU(),
                        fits.ImageHDU(np.array(k_modes_values, dtype=int), name="K_MODES_VALUES")
                    ])
                    for i in range(n_inputs):
                        ext = f"RESID_{i:02}"
                        model_ext = f"MODEL_RESID_{i:02}"
                        sci_array_stack = np.stack(sci_arrays_by_output[i])
                        model_array_stack = np.stack(model_arrays_by_output[i])
                        resid_hdul.append(fits.ImageHDU(sci_array_stack, name=ext))
                        resid_hdul.append(fits.ImageHDU(model_array_stack, name=model_ext))
                    with destination.open_path('residuals.fits', 'wb') as fh:
                        resid_hdul.writeto(fh)
            if self.save_inputs:
                inputs_hdul = fits.HDUList([
                    fits.PrimaryHDU(),
                ])
                for i in range(n_inputs):
                    ext = f"INPUT_{i:02}"
                    model_ext = f"MODEL_INPUT_{i:02}"
                    inputs_hdul.append(fits.ImageHDU(res.subtraction_result.pipeline_inputs[i].sci_arr, name=ext))
                    inputs_hdul.append(fits.ImageHDU(res.subtraction_result.pipeline_inputs[i].model_arr, name=model_ext))
                with destination.open_path('inputs.fits', 'wb') as fh:
                    inputs_hdul.writeto(fh)
        return res