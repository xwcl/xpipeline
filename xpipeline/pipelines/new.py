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
        if cache and self._cache is not None:
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
            scale_factors=self.scale_factors.load()
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
    k_modes : int = xconf.field(default=5, help="")
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
        image_vecs: np.ndarray,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        probe_model_vecs: Optional[np.ndarray] = None,
        decomposition: Optional[learning.PrecomputedDecomposition] = None,
    ):
        params = starlight_subtraction.KlipParams(
            k_klip=k_modes,
            exclusions=self._make_exclusions(self.exclude, angles),
            decomposer=self.decomposer.to_callable(),
            initial_decomposition_only=True,
            reuse=self.reuse,
        )
        return starlight_subtraction.klip_mtx(image_vecs, params=params)

    def execute(
        self, 
        image_vecs: np.ndarray,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        probe_model_vecs: Optional[np.ndarray] = None,
        decomposition: Optional[learning.PrecomputedDecomposition] = None,
    ):
        params = starlight_subtraction.KlipParams(
            k_klip=k_modes,
            exclusions=self._make_exclusions(self.exclude, angles),
            decomposer=self.decomposer.to_callable(),
            initial_decomposition=decomposition,
            reuse=self.reuse,
        )
        return starlight_subtraction.klip_mtx(image_vecs, params=params)

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
        image_vecs: np.ndarray,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        probe_model_vecs: Optional[np.ndarray] = None,
        decomposition: Optional[learning.PrecomputedDecomposition] = None,
    ) -> learning.PrecomputedDecomposition:
        klipt_params = starlight_subtraction.KlipTParams(
            k_modes=k_modes,
            **self.construct_klipt_params_dict()
        )
        return starlight_subtraction.compute_klipt_basis(image_vecs, probe_model_vecs, klipt_params)

    def execute(
        self,
        image_vecs: np.ndarray,
        k_modes : int,
        *,
        angles : Optional[np.ndarray] = None,
        probe_model_vecs: Optional[np.ndarray] = None,
        decomposition: Optional[learning.PrecomputedDecomposition] = None,
    ):
        params_kt = starlight_subtraction.KlipTParams(
            k_modes=k_modes,
            **self.construct_klipt_params_dict()
        )
        return starlight_subtraction.klip_transpose(
            image_vecs, decomposition, params_kt
        )



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
        signal = improc.wrap_matrix(
            signal_mtx[start_idx:end_idx],
            input_data.estimation_mask,
            fill_value=fill_value,
        )

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
    companion : CompanionConfig = xconf.field(help="Companion amplitude and location to inject (scale 0 for no injection) and probe")

    def load(self) -> StarlightSubtractionData:
        angles = self.angles.load()[::self.decimate_frames_by]
        companion = self.companion.to_companionspec()
        model_gen_sec = 0
        pipeline_inputs = []
        for pinputconfig in self.inputs:
            if pinputconfig.model_inputs is None:
                raise ValueError(f"Pipeline input has no model information")
            pinput = pinputconfig.load()
            pinput.sci_arr = pinput.sci_arr[::self.decimate_frames_by]
            pinput.sci_arr = pinput.sci_arr / pinput.model_inputs.scale_factors[::self.decimate_frames_by, np.newaxis, np.newaxis]
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
                pinput.sci_arr = pinput.sci_arr + companion.scale * pinput.model_arr
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
class StarlightSubtract:
    strategy : Union[KlipTranspose,Klip] = xconf.field(help="Strategy with which to estimate and subtract starlight")
    combine : constants.CombineOperation = xconf.field(default=constants.CombineOperation.MEAN, help="How to combine image for stacking")
    return_residuals : bool = xconf.field(default=True, help="Whether residual images after starlight subtraction should be returned")
    return_inputs : bool = xconf.field(default=True, help="Whether original images before starlight subtraction should be returned")
    subtract_mean : bool = xconf.field(default=True, help="Take a mean along the time axis and subtract from inputs")
    # pre_stack_filter : Optional[PreStackFilter] = xconf.field(help="Process after removing starlight and before stacking")
    # return_pre_stack_filtered : bool = xconf.field(default=True, help="Whether filtered images before stacking should be returned")
    k_modes_values : list[int] = xconf.field(default_factory=lambda: [12], help="Which values to try for number of modes to subtract")
    return_decomposition : bool = xconf.field(default=True, help="Whether the computed decomposition should be returned")

    def execute(self, data : StarlightSubtractionData) -> StarlightSubtractResult:
        destination_exts = defaultdict(list)
        for pinput in data.inputs:
            if len(destination_exts[pinput.destination_ext]):
                if pinput.sci_arr.shape != destination_exts[pinput.destination_ext][0].sci_arr.shape:
                    raise ValueError(f"Dimensions of current science array {pinput.sci_arr.shape=} mismatched with others. Use separate destination_ext settings for each input, or make them the same size.")
                if self.subtract_mean:
                    pinput.sci_arr = pinput.sci_arr - np.nanmean(pinput.sci_arr, axis=0)
            destination_exts[pinput.destination_ext].append(pinput)

        data_vecs, model_vecs = unwrap_inputs_to_matrices(data.inputs)
        print(f"{model_vecs.shape=}")

        decomp = data.initial_decomposition

        if decomp is None:
            max_k_modes = max(self.k_modes_values)
            log.debug(f"Computing basis with {max_k_modes=}")
            decomp = self.strategy.prepare(
                data_vecs,
                max_k_modes,
                angles=data.angles,
                probe_model_vecs=model_vecs,
            )

        results_for_modes = {}
        for mode_idx, k_modes in enumerate(self.k_modes_values):
            log.debug(f"Subtracting starlight for modes value k={k_modes} ({mode_idx+1}/{len(self.k_modes_values)})")
            data_vecs_resid = self.strategy.execute(
                data_vecs,
                k_modes,
                angles=data.angles,
                probe_model_vecs=model_vecs,
                decomposition=decomp,
            )
            model_vecs_resid = self.strategy.execute(
                model_vecs,
                k_modes,
                angles=data.angles,
                probe_model_vecs=model_vecs,
                decomposition=decomp,
            )
            
            pipeline_outputs = residuals_matrix_to_outputs(
                data_vecs_resid,
                data.inputs,
                mean_vec=None,
                signal_mtx=model_vecs_resid,
            )
            # filter individual frames before stacking
            # filtered_outputs = self.pre_stack_filter.execute(pipeline_outputs)

            # group outputs by their destination extension
            outputs_by_ext = defaultdict(list)
            for po in pipeline_outputs:
                outputs_by_ext[po.destination_ext].append(po)
            
            destination_images = {}
            for ext in outputs_by_ext:
                # construct a single outputs cube with all contributing inputs
                all_outputs_cube = None
                for po in outputs_by_ext[ext]:
                    assert isinstance(po, PipelineOutput)
                    derot_cube = improc.derotate_cube(po.sci_arr, data.angles)
                    if all_outputs_cube is None:
                        all_outputs_cube = derot_cube
                    else:
                        all_outputs_cube = np.concatenate([all_outputs_cube, derot_cube])
                # combine the concatenated cube into a single plane
                destination_images[ext] = improc.combine(all_outputs_cube, self.combine)

            results_for_modes[k_modes] = StarlightSubtractModesResult(
                destination_images=destination_images,
                pipeline_outputs=pipeline_outputs if self.return_residuals else None,
            )

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
    signal : float
    snr : float
    simple_signal: float
    simple_snr : float

# @xconf.config
# class  _BasePostFilter:
#     return_filter_kernel : bool = xconf.field(default=False)
#     return_filtered_image : bool = xconf.field(default=False)

#     def execute(
#         self,
#         destination_image: np.ndarray,
#         pipeline_outputs_for_ext: list[PipelineOutput],
#         measurement_location : CompanionSpec,
#         derotation_angles: Optional[np.ndarray] = None,
#     ) -> PostFilteringResult:
#         raise NotImplementedError("Subclasses must implement execute()")

# @xconf.config
# class TophatPostFilter(_BasePostFilter):
#     radius_px : float = xconf.field(default=4, help="Top-hat kernel radius")
#     exclude_nearest_apertures : int = xconf.field(default=1, help="Exclude this many apertures on either side of the measurement location from the noise sample")

#     def execute(
#         self,
#         destination_image: np.ndarray,
#         pipeline_outputs_for_ext: list[PipelineOutput],
#         measurement_location : CompanionSpec,
#         derotation_angles: Optional[np.ndarray] = None,
#     ) -> PostFilteringResult:
#         kernel = Tophat2DKernel(self.radius_px)
#         filtered_image = convolve_fft(
#             destination_image,
#             kernel
#         )
#         snr, signal = snr_from_convolution(
#             filtered_image,
#             measurement_location.r_px,
#             measurement_location.pa_deg,
#             self.radius_px*2,
#             exclude_nearest=self.exclude_nearest_apertures
#         )
#         return PostFilteringResult(
#             signal=signal,
#             snr=snr,
#             filter_kernel=kernel.array if self.return_filter_kernel else None,
#             filtered_image=filtered_image if self.return_filtered_image else None,
#         )

def signal_from_filter(img, mf):
    mf /= np.nansum(mf**2)
    mf = np.flip(mf, axis=(0, 1))
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
    by_ext : dict[str, PostFilteringResult]

@dataclass
class StarlightSubtractionMeasurements:
    companion : CompanionSpec
    by_modes: dict[int,StarlightSubtractionMeasurement]
    subtraction_result : Optional[StarlightSubtractResult]
      

@xconf.config
class MeasureStarlightSubtraction:
    resolution_element_px : float = xconf.field(help="Diameter of the resolution element in pixels")
    exclude_nearest_apertures: int = xconf.field(default=1, help="How many locations on either side of the probed companion location should be excluded from the SNR calculation")
    subtraction : StarlightSubtract = xconf.field(help="Configure starlight subtraction options")
    return_starlight_subtraction : bool = xconf.field(default=False, help="whether to return the starlight subtraction result")

    def __post_init__(self):
        # self.subtraction.return_residuals = True
        self.subtraction.return_decomposition = True

    def execute(
        self, data : StarlightSubtractionData
    ) -> list[StarlightSubtractionMeasurement]:
        ssresult : StarlightSubtractResult = self.subtraction.execute(data)
        destination_exts = defaultdict(list)
        for pinput in data.inputs:
            destination_exts[pinput.destination_ext].append(pinput)
        # dest_exts = tuple(ssresult.modes[self.subtraction.k_modes_values[0]].destination_images.keys())
        r_px, pa_deg = data.companions[0].r_px, data.companions[0].pa_deg
        
        iwa_px = r_px - self.resolution_element_px / 2
        companionspecs = list(characterization.generate_probes(
            iwa_px=iwa_px,
            owa_px=iwa_px + self.resolution_element_px,
            n_radii=1,
            spacing_px=self.resolution_element_px,
            starting_pa_deg=pa_deg,
            scales=[0.0]
        ))
        measurement_loc = companionspecs[0]
        companionspecs = [measurement_loc,] + companionspecs[1 + self.exclude_nearest_apertures:-self.exclude_nearest_apertures]
        
        result = StarlightSubtractionMeasurements(data.companions[0], {}, ssresult if self.return_starlight_subtraction else None)
        for k_modes in self.subtraction.k_modes_values:
            result.by_modes[k_modes] = StarlightSubtractionMeasurement(by_ext={})

        for dest_ext in destination_exts:
            signal, noises = 0, []
            log.debug(f"Measuring signal with circular aperture")
            for k_modes in self.subtraction.k_modes_values:
                img = ssresult.modes[k_modes].destination_images[dest_ext]
                kernel = Tophat2DKernel(self.resolution_element_px / 2)
                filtered_image = convolve_fft(
                    img,
                    kernel
                )
                simple_snr, simple_signal = snr_from_convolution(
                    filtered_image,
                    r_px,
                    pa_deg,
                    self.resolution_element_px,
                    exclude_nearest=self.exclude_nearest_apertures
                )
                result.by_modes[k_modes].by_ext[dest_ext] = PostFilteringResult(
                    signal=None, snr=None, ## todo make this less hairy somehow
                    simple_signal=simple_signal,
                    simple_snr=simple_snr,
                )
            signals_for_modes = {}
            noises_for_modes = {}
            for k_modes in self.subtraction.k_modes_values:
                noises_for_modes[k_modes] = []
            for cidx, companionspec in enumerate(companionspecs):
                model_data = StarlightSubtractionData(
                    inputs=[],
                    companions=[],
                    angles=data.angles,
                    initial_decomposition=ssresult.decomposition
                )
                probe_ext = f"probe_{cidx}"
                for pinput in destination_exts[dest_ext]:
                    log.debug(f"Generating probe {probe_ext}")
                    model_arr = generate_signal(
                        pinput.sci_arr.shape,
                        companionspec.r_px,
                        companionspec.pa_deg,
                        pinput.model_inputs.arr,
                        data.angles,
                    )
                    model_data.inputs.append(PipelineInput(
                        model_arr,
                        pinput.estimation_mask,
                        destination_ext=probe_ext,
                        combination_mask=pinput.combination_mask,
                        model_inputs=pinput.model_inputs,
                        model_arr=pinput.model_arr
                    ))
                model_ss_result = self.subtraction.execute(model_data)
                for k_modes in ssresult.modes:
                    kernel_for_loc = model_ss_result.modes[k_modes].destination_images[probe_ext]
                    val = signal_from_filter(img, kernel_for_loc)
                    if cidx == 0:
                        log.debug(f"measuring signal using {probe_ext}")
                        signals_for_modes[k_modes] = val
                    else:
                        noises_for_modes[k_modes].append(val)
            for k_modes in ssresult.modes:
                snr = characterization.calc_snr_mawet(signals_for_modes[k_modes], noises_for_modes[k_modes])
                result.by_modes[k_modes].by_ext[dest_ext].signal = signal
                result.by_modes[k_modes].by_ext[dest_ext].snr = snr
        return result

@xconf.config
class MeasureStarlightSubtractionPipeline(MeasureStarlightSubtraction):
    data : StarlightSubtractionDataConfig = xconf.field(help="Starlight subtraction data")
    def execute(self) -> StarlightSubtractionMeasurements:
        data = self.data.load()
        return super().execute(data)