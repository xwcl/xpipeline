from collections import defaultdict
from astropy.convolution import convolve_fft, Tophat2DKernel
import math
import time
from copy import copy, deepcopy
from dataclasses import dataclass
import xconf
import numpy as np
from xconf.contrib import BaseRayGrid, FileConfig, join, PathConfig, DirectoryConfig
import sys
import logging
from typing import Optional, Union
from ..commands.base import BaseCommand
from .. import utils
from ..tasks import starlight_subtraction, learning, improc, characterization
from ..tasks.characterization import CompanionSpec, r_pa_to_x_y, snr_from_convolution, calculate_snr
from .. import constants
from xpipeline.types import FITS_EXT
import logging

log = logging.getLogger(__name__)


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

    def load(self, cache=True) -> np.ndarray:
        if cache and getattr(self, '_cache', None) is not None:
            return self._cache
        from ..tasks import iofits
        with self.open() as fh:
            hdul = iofits.load_fits(fh)
        data = hdul[self.ext].data
        if cache:
            self._cache = data
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
        if cache and getattr(self, '_cache', None) is not None:
            return self._cache
        from ..tasks import iofits
        with self.open() as fh:
            hdul = iofits.load_fits(fh)
        coldata = hdul[self.ext].data[self.table_column]
        if cache:
            self._cache = coldata
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
    destination_ext: str = "finim"
    combination_mask: Optional[np.ndarray] = None
    model_inputs : Optional[ModelSignalInput] = None
    model_arr : Optional[np.ndarray] = None

@dataclass
class StarlightSubtractionData:
    inputs : list[PipelineInput]
    angles : np.ndarray
    initial_decomposition : learning.PrecomputedDecomposition
    companions : list[CompanionSpec]

    def max_rank(self):
        cols = 0
        rows = 0
        for pi in self.inputs:
            cols += pi.sci_arr.shape[0]
            rows += np.count_nonzero(pi.estimation_mask)
        return min(rows, cols)

@dataclass
class PipelineOutput:
    sci_arr: np.ndarray
    destination_ext: str = "finim"
    model_arr: Optional[np.ndarray] = None
    mean_image: Optional[np.ndarray] = None

@xconf.config
class PipelineInputConfig:
    sci_arr: FitsConfig = xconf.field(help="Science frames as cube")
    estimation_mask: Optional[FitsConfig] = xconf.field(default=None, help="Estimation mask with the shape of a single plane of sci_arr")
    combination_mask: Optional[FitsConfig] = xconf.field(default=None, help="Combination mask with the shape of a single plane of sci_arr, or False to exclude from final combination")
    radial_mask : Optional[RadialMaskConfig] = xconf.field(default=None, help="Radial mask to exclude pixels min_r_px > r || r > max_r_px from center")
    model_inputs : Optional[ModelSignalInputConfig] = xconf.field(default=None, help="Model signal for matched filtering")
    destination_ext: str = xconf.field(default="finim", help="Extension into which final image should be combined")

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
            destination_ext=self.destination_ext,
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
    reuse : bool = xconf.field(default=True, help="Use the same basis set for all frames")
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
        # image_vecs: np.ndarray,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        # probe_model_vecs: Optional[np.ndarray] = None,
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
        # image_vecs: np.ndarray,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        # probe_model_vecs: Optional[np.ndarray] = None,
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
        return starlight_subtraction.klip_mtx(image_vecs, params=params, probe_model_vecs=probe_model_vecs)

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
        # probe_model_vecs: Optional[np.ndarray] = None,
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
        # probe_model_vecs: Optional[np.ndarray] = None,
        decomposition: Optional[learning.PrecomputedDecomposition] = None,
    ):
        image_vecs, probe_model_vecs = unwrap_inputs_to_matrices(data.inputs)
        image_vecs_medsub = image_vecs - np.median(image_vecs, axis=0)
        mtx_u = decomposition.mtx_u0[:,:k_modes]
        diag_s = decomposition.diag_s0[:k_modes]
        mtx_v = decomposition.mtx_v0[:,:k_modes]
        subspace_image_vec_projections = (mtx_u * diag_s) @ mtx_v.T
        return image_vecs_medsub - subspace_image_vec_projections

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
        # image_vecs: np.ndarray,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        # probe_model_vecs: Optional[np.ndarray] = None,
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
        # image_vecs: np.ndarray,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        # probe_model_vecs: Optional[np.ndarray] = None,
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
        return image_vecs_resid, model_vecs_resid, decomposition, med_vec



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
        pipeline_outputs.append(PipelineOutput(cube, input_data.destination_ext, signal, mean_image))
        start_idx += n_features
    return pipeline_outputs

@dataclass
class StarlightSubtractModesResult:
    destination_images : dict[str, np.ndarray]
    pipeline_outputs : Optional[list[PipelineOutput]]

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
class StarlightSubtractionDataConfig:
    inputs : list[PipelineInputConfig] = xconf.field(help="Input data to simultaneously reduce")
    angles : Union[FitsConfig,FitsTableColumnConfig] = xconf.field(help="1-D array or table column of derotation angles")
    initial_decomposition : Optional[PrecomputedDecompositionConfig] = xconf.field(default=None, help="Initial decomposition of the data to reuse")
    decimate_frames_by : int = xconf.field(default=1, help="Keep every Nth frame")
    decimate_frames_offset : int = xconf.field(default=0, help="Slice to begin decimation at this frame")
    companion : CompanionConfig = xconf.field(default=CompanionConfig(r_px=30, pa_deg=0, scale=0), help="Companion amplitude and location to inject (scale 0 for no injection) and probe")

    def load(self) -> StarlightSubtractionData:
        angles = self.angles.load()[self.decimate_frames_offset::self.decimate_frames_by]
        companion = self.companion.to_companionspec()
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
            pinput.model_arr = generate_signal(
                pinput.sci_arr.shape,
                companion.r_px,
                companion.pa_deg,
                pinput.model_inputs.arr,
                angles,
                # pinput.model_inputs.scale_factors[::self.decimate_frames_by],
            )
            dt = time.time() - ts
            model_gen_sec += dt
            if companion.scale != 0:
                pinput.sci_arr += companion.scale * pinput.model_arr
            pipeline_inputs.append(pinput)
        log.debug("Spent %f seconds in model generation", model_gen_sec)

        if self.initial_decomposition is not None:
            initial_decomposition = self.initial_decomposition.load()
        else:
            initial_decomposition = None
        return StarlightSubtractionData(
            inputs=pipeline_inputs,
            angles=angles,
            initial_decomposition=initial_decomposition,
            companions=[companion]
        )

@xconf.config
class KModesValuesConfig:
    values: list[int] = xconf.field(help="Which values to try for number of modes to subtract")

    def as_values(self, max_rank: int):
        values = [x for x in self.values if x < max_rank]
        if not len(values):
            raise ValueError(f"Given {max_rank=}, no valid values from {self.values}")
        return values

@xconf.config
class KModesFractionConfig:
    fractions: list[float] = xconf.field(default_factory=lambda: [0.1], help="Which values to try for number of modes to subtract")

    def as_values(self, max_rank: int):
        if any(x >= 1.0 for x in self.fractions):
            raise ValueError(f"Invalid fractions in config: {self.fractions} (must be 0 < x < 1)")
        values = [int(x * max_rank) for x in self.fractions]
        return values

KModesConfig = Union[KModesValuesConfig,KModesFractionConfig]

@xconf.config
class PreStackFilter:
    def execute(
        self,
        pipeline_outputs: list[PipelineOutput],
        measurement_location: CompanionSpec,
        resolution_element_px : float,
        derotation_angles: Optional[np.ndarray]=None,
    ) -> list[PipelineOutput]:
        out = []
        for po in pipeline_outputs:
            out.append(PipelineOutput(po.sci_arr, po.destination_ext, model_arr=po.model_arr))
        return out


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
        for po in pipeline_outputs:
            sci_arr_filtered = np.zeros_like(po.sci_arr)
            for i in range(po.sci_arr.shape[0]):
                radius_px = resolution_element_px
                kernel = Tophat2DKernel(radius=radius_px)
                sci_arr_filtered[i] = convolve_fft(
                    po.sci_arr[i],
                    kernel,
                    nan_treatment='fill',
                    fill_value=0.0,
                    preserve_nan=True,
                )
            out.append(PipelineOutput(sci_arr_filtered, po.destination_ext, model_arr=po.model_arr))
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
        pipeline_outputs[0].model_arr
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
            out.append(PipelineOutput(sci_arr_filtered, po.destination_ext, model_arr=po.model_arr))
        return out

@xconf.config
class ImageStack:
    combine : constants.CombineOperation = xconf.field(default=constants.CombineOperation.MEDIAN, help="How to combine image for stacking")
    minimum_coverage_frac: float = xconf.field(default=0.2, help="Number of overlapping source frames covering a derotated frame pixel for it to be kept in the final image as a fraction of the total number of frames")
    return_residuals : bool = xconf.field(default=True, help="Whether residual images after starlight subtraction should be returned")

    def execute(self, pipeline_outputs: list[PipelineOutput], angles: Optional[np.ndarray]):
        # group outputs by their destination extension
        outputs_by_ext = defaultdict(list)
        for po in pipeline_outputs:
            outputs_by_ext[po.destination_ext].append(po)
        destination_images = {}
        for ext in outputs_by_ext:
            log.debug(f"Stacking {len(outputs_by_ext[ext])} outputs for {ext=}")
            # construct a single outputs cube with all contributing inputs
            all_outputs_cube = None
            mask_good = ~(np.nanmin(po.sci_arr, axis=(1,2)) == np.nanmax(po.sci_arr, axis=(1,2)))
            for po in outputs_by_ext[ext]:
                assert isinstance(po, PipelineOutput)
                derot_cube = improc.derotate_cube(po.sci_arr[mask_good], angles[mask_good])
                if all_outputs_cube is None:
                    all_outputs_cube = derot_cube
                else:
                    all_outputs_cube = np.concatenate([all_outputs_cube, derot_cube])
            # combine the concatenated cube into a single plane
            finim = improc.combine(all_outputs_cube, self.combine)
            # apply minimum coverage mask
            finite_elements_cube = np.isfinite(all_outputs_cube)
            coverage_count = np.sum(finite_elements_cube, axis=0)
            coverage_mask = coverage_count > (finite_elements_cube.shape[0] * self.minimum_coverage_frac)
            finim[~coverage_mask] = np.nan
            destination_images[ext] = finim

        return StarlightSubtractModesResult(
            destination_images=destination_images,
            pipeline_outputs=pipeline_outputs if self.return_residuals else None,
        )

@xconf.config
class StarlightSubtract:
    strategy : Union[KlipTranspose,Klip,KlipSubspace] = xconf.field(help="Strategy with which to estimate and subtract starlight")
    resolution_element_px : float = xconf.field(help="One resolution element (lambda / D) in pixels")
    return_inputs : bool = xconf.field(default=True, help="Whether original images before starlight subtraction should be returned")
    image_stack: ImageStack = xconf.field(default=ImageStack(), help="How to combine images after starlight subtraction and filtering")
    pre_stack_filter : Union[MatchedPreStackFilter,TophatPreStackFilter,NoOpPreStackFilter] = xconf.field(default=None, help="Process after removing starlight and before stacking")
    # return_pre_stack_filtered : bool = xconf.field(default=True, help="Whether filtered images before stacking should be returned")
    k_modes : KModesConfig = xconf.field(default_factory=KModesFractionConfig, help="Which values to try for number of modes to subtract")
    return_decomposition : bool = xconf.field(default=True, help="Whether the computed decomposition should be returned")

    def execute(self, data : StarlightSubtractionData) -> StarlightSubtractResult:
        destination_exts = defaultdict(list)
        for pinput in data.inputs:
            if len(destination_exts[pinput.destination_ext]):
                if pinput.sci_arr.shape != destination_exts[pinput.destination_ext][0].sci_arr.shape:
                    raise ValueError(f"Dimensions of current science array {pinput.sci_arr.shape=} mismatched with others. Use separate destination_ext settings for each input, or make them the same size.")
            destination_exts[pinput.destination_ext].append(pinput)
        max_rank = data.max_rank()
        k_modes_values = self.k_modes.as_values(max_rank)
        log.debug(f"Estimation masks and data cubes imply max rank {max_rank}, implying {k_modes_values=} from {self.k_modes}")
        decomp = data.initial_decomposition

        if decomp is None:
            max_k_modes = max(k_modes_values)
            log.debug(f"Computing basis with {max_k_modes=}")
            decomp = self.strategy.prepare(
                data,
                max_k_modes,
                angles=data.angles,
            )

        results_for_modes = {}
        for mode_idx, k_modes in enumerate(k_modes_values):
            log.debug(f"Subtracting starlight for modes value k={k_modes} ({mode_idx+1}/{len(k_modes_values)})")
            res = self.strategy.execute(
                data,
                k_modes,
                angles=data.angles,
                decomposition=decomp,
            )
            data_vecs_resid, model_vecs_resid, _, _ = res
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
                    data.angles
                )
            else:
                filtered_outputs = pipeline_outputs
            results_for_modes[k_modes] = self.image_stack.execute(filtered_outputs, data.angles)

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
class PostFilteringResult:
    image : np.ndarray
    kernel_diameter_px : Union[float,int]
    kernel : Optional[np.ndarray] = None

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
class NoPostFilter(_BasePostFilter):
    def execute(
        self,
        destination_image: np.ndarray,
        pipeline_outputs_for_ext: list[PipelineOutput],
        measurement_location : CompanionSpec,
        resolution_element_px : float,
        derotation_angles: Optional[np.ndarray] = None,
    ) -> PostFilteringResult:
        return PostFilteringResult(
            kernel=np.ones((1,1)),
            image=destination_image,
            kernel_diameter_px=resolution_element_px
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
        for idx, output in enumerate(pipeline_outputs_for_ext):
            log.debug(f"Derotating model residuals from output {idx + 1} / {len(pipeline_outputs_for_ext)}")
            derotated_output = improc.derotate_cube(output.model_arr, derotation_angles)
            if derotated_cube is None:
                derotated_cube = derotated_output
            else:
                derotated_cube = np.concatenate([derotated_cube, derotated_output])
        kernel = improc.combine(derotated_cube, self.combine)
        
        # shift kernel to center
        dx, dy = characterization.r_pa_to_x_y(measurement_location.r_px, measurement_location.pa_deg, 0, 0)
        kernel = improc.shift2(kernel, -dx, -dy)

        # trim kernel to only central region
        radius_px = (self.kernel_diameter_resel * resolution_element_px) / 2
        log.debug(f"Matched filter with {self.kernel_diameter_resel} lambda/D extent is {radius_px=} given {resolution_element_px=}")
        rho, _ = improc.polar_coords(improc.arr_center(kernel), kernel.shape)
        kernel[rho > radius_px] = 0

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

@dataclass
class StarlightSubtractionMeasurement:
    signal : float
    snr : float
    post_filtering_result : Optional[PostFilteringResult]

@dataclass
class StarlightSubtractionMeasurementSet:
    by_ext : dict[str, dict[str, StarlightSubtractionMeasurement]]

@dataclass
class StarlightSubtractionMeasurements:
    companion : CompanionSpec
    by_modes: dict[int,StarlightSubtractionMeasurementSet]
    subtraction_result : Optional[StarlightSubtractResult]

PostFilter = Union[TophatPostFilter, MatchedPostFilter]

@xconf.config
class MeasureStarlightSubtraction:
    # resolution_element_px : float = xconf.field(help="Diameter of the resolution element in pixels")
    exclude_nearest_apertures: int = xconf.field(default=1, help="How many locations on either side of the probed companion location should be excluded from the SNR calculation")
    subtraction : StarlightSubtract = xconf.field(help="Configure starlight subtraction options")
    return_starlight_subtraction : bool = xconf.field(default=True, help="whether to return the starlight subtraction result")
    tophat_post_filter : TophatPostFilter = xconf.field(default=TophatPostFilter(), help="Filter final derotated images with a circular aperture")
    matched_post_filter : Optional[MatchedPostFilter] = xconf.field(default=MatchedPostFilter(), help="Filter final derotated images with a matched filter based on the model PSF")
    return_post_filtering_result : bool = xconf.field(default=True, help="whether to return the images and kernels from filtering")

    def __post_init__(self):
        self.subtraction.return_residuals = self.matched_post_filter is not None
        self.subtraction.return_decomposition = True

    def execute(
        self, data : StarlightSubtractionData
    ) -> StarlightSubtractionMeasurements:
        ssresult : StarlightSubtractResult = self.subtraction.execute(data)
        companion = data.companions[0]
        result = StarlightSubtractionMeasurements(
            companion=companion,
            by_modes={},
            subtraction_result=ssresult if self.return_starlight_subtraction else None,
        )
        for k_modes in ssresult.modes:
            outputs_for_ext = defaultdict(list)
            for poutput in ssresult.modes[k_modes].pipeline_outputs:
                outputs_for_ext[poutput.destination_ext].append(poutput)    
            meas = StarlightSubtractionMeasurementSet(by_ext={})
            for ext in ssresult.modes[k_modes].destination_images:
                image = ssresult.modes[k_modes].destination_images[ext]
                post_filters = {
                    'tophat': self.tophat_post_filter,
                    'none': NoPostFilter(),
                }
                if self.matched_post_filter is not None:
                    post_filters['matched'] = self.matched_post_filter
                filter_meas = {}
                for name, post_filter in post_filters.items():
                    if post_filter is None:
                        continue
                    filter_result = post_filter.execute(
                        image,
                        outputs_for_ext[ext],
                        companion,
                        self.subtraction.resolution_element_px,
                        data.angles,
                    )
                    snr, signal = snr_from_convolution(
                        filter_result.image,
                        loc_rho=companion.r_px,
                        loc_pa_deg=companion.pa_deg,
                        aperture_diameter_px=filter_result.kernel_diameter_px,
                        exclude_nearest=self.exclude_nearest_apertures,
                        good_pixel_mask=np.isfinite(filter_result.image),
                    )

                    filter_meas[name] = StarlightSubtractionMeasurement(
                        signal=signal,
                        snr=snr,
                        post_filtering_result=filter_result if self.return_post_filtering_result else None,
                    )
                meas.by_ext[ext] = filter_meas
            result.by_modes[k_modes] = meas
        return result

    def measurements_to_jsonable(self, res, k_modes_values):
        output_dict = {}
        output_dict['config'] = xconf.asdict(self)
        output_dict['results'] = {
            'k_modes_values': k_modes_values,
        }

        for k in k_modes_values:
            for ext in res.by_modes[k].by_ext:
                if ext not in output_dict['results']:
                    output_dict['results'][ext] = []
                filtered_measurements = {}
                for filter_name in res.by_modes[k].by_ext[ext]:
                    filtered_measurements[filter_name] = {
                        'snr': res.by_modes[k].by_ext[ext][filter_name].snr,
                        'signal': res.by_modes[k].by_ext[ext][filter_name].signal,
                    }
                output_dict['results'][ext].append(filtered_measurements)
        return output_dict

@xconf.config
class MeasureStarlightSubtractionPipeline(MeasureStarlightSubtraction):
    data : StarlightSubtractionDataConfig = xconf.field(help="Starlight subtraction data")
    def execute(self) -> StarlightSubtractionMeasurements:
        data = self.data.load()
        return super().execute(data)