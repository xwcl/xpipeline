import logging
from pprint import pformat
import numpy as np
import typing
import dask
import dask.array as da
import pandas as pd
from typing import List, Union, Optional

from .. import constants as const
from .. import core
from ..core import PipelineCollection, reduce_bitwise_or

from ..tasks import (
    obs_table,
    iofits,
    sky_model,
    detector,
    data_quality,
    learning,
    improc,
    starlight_subtraction,
    characterization,
    vapp,
)
from ..tasks.starlight_subtraction import KlipInput, KlipParams
from ..constants import KlipStrategy, CombineOperation
from ..ref import clio

log = logging.getLogger(__name__)

# To preserve sanity, pipelines mustn't do anything you can't do to a dask.delayed
# and mustn't compute or branch on intermediate results.
# They should return delayeds (for composition)


def combine_extension_to_new_hdu(
    inputs_collection : PipelineCollection,
    operation : CombineOperation,
    ext : typing.Union[str, int],
    plane_shape : tuple[int, int]
):
    image_cube = iofits.hdulists_to_dask_cube(inputs_collection.items, plane_shape, ext=ext)
    if operation is CombineOperation.MEAN:
        result = da.nanmean(image_cube, axis=0)
    if not isinstance(ext, int):
        hdr = {'EXTNAME': ext}
    else:
        hdr = None
    return dask.delayed(iofits.DaskHDU)(result, header=hdr, kind="image")


def compute_scale_factors(
    inputs_collection_or_hdul : typing.Union[PipelineCollection,iofits.DaskHDUList],
    template_hdul : iofits.DaskHDUList,
    saturated_pixel_threshold : float,
):
    delayed_hdus = []
    for extname in template_hdul.extnames:
        if template_hdul[extname].data is None or len(template_hdul[extname].data.shape) != 2:
            continue
        plane_shape = template_hdul[extname].data.shape
        log.debug(f'{plane_shape=}')
        if isinstance(inputs_collection_or_hdul, iofits.DaskHDUList):
            data_cube = da.from_array(inputs_collection_or_hdul[extname].data)
        else:
            data_cube = iofits.hdulists_to_dask_cube(inputs_collection_or_hdul.items, plane_shape, ext=extname)
        template_array = template_hdul[extname].data
        radii, profile = improc.trim_radial_profile(template_array)
        d_factors = dask.delayed(np.asarray)([
            dask.delayed(improc.template_scale_factor_from_image)(x, radii, profile, saturated_pixel_threshold=saturated_pixel_threshold)
            for x in data_cube
        ])
        def _to_hdu(data, name=None):
            return iofits.DaskHDU(data, name=name)
        delayed_hdus.append(dask.delayed(_to_hdu)(d_factors, name=extname if not isinstance(extname, int) else None))

    def _to_hdulist(*args):
        hdus = [iofits.DaskHDU(data=None, kind="primary")]
        hdus.extend(args)
        return iofits.DaskHDUList(hdus)
    return dask.delayed(_to_hdulist)(*delayed_hdus)


def clio_split(
    inputs_coll: PipelineCollection,
    inputs_filenames: List[str],
    frames_per_cube,
    ext=0,
) -> PipelineCollection:
    log.debug("Assembling clio_split pipeline...")
    n_inputs = len(inputs_coll.items)
    outputs = inputs_coll.collect(
        clio.serial_split_frames_cube,
        inputs_filenames,
        _delayed_kwargs={"nout": n_inputs * frames_per_cube},
    )
    log.debug("done assembling clio_split")
    return inputs_coll.with_new_contents(outputs)


def clio_badpix_linearity(
    inputs_coll: PipelineCollection,
    badpix_arr: np.ndarray,
    plane_shape: tuple[int, int],
    ext=0,
) -> PipelineCollection:
    log.debug("Assembling clio_badpix_linearity pipeline...")
    coll = (
        inputs_coll.map(iofits.ensure_dq)
        .map(data_quality.set_dq_flag, badpix_arr, const.DQ_BAD_PIXEL)
        .map(clio.correct_linearity)
    )
    log.debug("done assembling clio_badpix_linearity")
    return coll


def sky_subtract(
    input_coll: PipelineCollection,
    model_sky: sky_model.SkyModel,
    mask_dilate_iters: int,
    n_sigma: float,
    ext=0,
    dq_ext='DQ',
    excluded_pixels_mask: Optional[np.ndarray] = None,
):
    coll = input_coll.map(
        sky_model.background_subtract,
        model_sky,
        mask_dilate_iters,
        n_sigma=n_sigma,
        ext=ext,
        dq_ext=dq_ext,
        excluded_pixels_mask=excluded_pixels_mask,
    )
    return coll


def align_to_templates(
    input_coll: PipelineCollection,
    cutout_specs: List[improc.CutoutTemplateSpec],
    *,
    upsample_factor: int = 100,
    ext: Union[int, str] = 0,
    dq_ext: Union[int, str] = "DQ",
    excluded_pixels_mask = None,
) -> PipelineCollection:
    log.debug(f'align_to_templates {cutout_specs=}')
    # explode list of cutout_specs into individual cutout pipelines
    d_hdus_for_cutouts = []
    for cspec in cutout_specs:
        d_hdus = (input_coll
                  .map(data_quality.get_masked_data, ext=ext, dq_ext=dq_ext, permitted_flags=const.DQ_SATURATED, excluded_pixels_mask=excluded_pixels_mask)
                  .map(improc.aligned_cutout, cspec, upsample_factor=upsample_factor)
                  .map(iofits.DaskHDU, header={'EXTNAME': cspec.name})
                  .items
                  )
        d_hdus_for_cutouts.append(d_hdus)

    # collect as multi-extension FITS
    def _collect(primary_hdu: iofits.DaskHDU, *args: iofits.DaskHDU):
        new_primary_hdu = primary_hdu.updated_copy(
            None,
            history="Detached header from full frame in conversion to multi-extension file",
        )
        new_primary_hdu.kind = 'primary'
        hdus = [new_primary_hdu]
        hdus.extend(args)
        return iofits.DaskHDUList(hdus)

    return input_coll.map(lambda x: x[ext]).zip_map(_collect, *d_hdus_for_cutouts)


def compute_sky_model(
    inputs_collection: PipelineCollection,
    plane_shape: tuple,
    test_fraction,
    random_state,
    n_components,
    mask_dilate_iters,
    ext: Union[int, str] = 0,
    dq_ext: Union[int, str] = 'DQ',
    excluded_pixels_mask : Optional[np.ndarray] = None,
):
    log.debug("Assembling compute_sky_model pipeline...")
    sky_cube = iofits.hdulists_to_dask_cube(inputs_collection.items, plane_shape, ext=ext)
    dq_cube = iofits.hdulists_to_dask_cube(inputs_collection.items, plane_shape, ext=dq_ext, dtype=int)
    badpix_arr = reduce_bitwise_or(dq_cube)
    if excluded_pixels_mask is not None:
        badpix_arr = badpix_arr | excluded_pixels_mask
    sky_cube_train, sky_cube_test = learning.train_test_split(
        sky_cube, test_fraction, random_state=random_state
    )
    components, mean_sky, stddev_sky = sky_model.compute_components(
        sky_cube_train, n_components
    )
    min_err, max_err, avg_err = dask.delayed(sky_model.cross_validate, nout=3)(
        sky_cube_test, components, stddev_sky, mean_sky, badpix_arr, mask_dilate_iters
    )
    log.debug("done assembling compute_sky_model")
    model = sky_model.SkyModel(
        components, mean_sky, stddev_sky, min_err, max_err, avg_err
    )
    return model

def klip_inputs_to_mtx_x(klip_inputs: List[KlipInput]):
    matrices = []
    signal_matrices = []
    xp = core.get_array_module(klip_inputs[0].sci_arr)
    for idx, input_data in enumerate(klip_inputs):
        mtx_x = improc.unwrap_cube(
            input_data.sci_arr, input_data.estimation_mask
        )
        if input_data.signal_arr is not None:
            mtx_x_signal_only = improc.unwrap_cube(
                input_data.signal_arr, input_data.estimation_mask
            )
        else:
            mtx_x_signal_only = None
        log.debug(
            f"klip input {idx} has {mtx_x.shape=} from {input_data.sci_arr.shape=} and "
            f"{np.count_nonzero(input_data.estimation_mask)=}"
        )
        matrices.append(mtx_x)
        signal_matrices.append(mtx_x_signal_only)

    mtx_x = xp.vstack(matrices)
    if all([ki.signal_arr is not None for ki in klip_inputs]):
        mtx_x_signal_only = xp.vstack(signal_matrices)
    elif any([ki.signal_arr is not None for ki in klip_inputs]):
        raise ValueError("Some inputs have signal arrays, some don't")
    else:
        mtx_x_signal_only = None
    return mtx_x, mtx_x_signal_only

def klip_multi(klip_inputs: List[KlipInput], klip_params: KlipParams):
    log.debug("assembling klip_multi")
    mtx_x, mtx_x_signal_only = klip_inputs_to_mtx_x(klip_inputs)
    
    # where the klipping happens
    subtracted_mtx, signal_mtx, decomposition, mean_vec = starlight_subtraction.klip_mtx(
        mtx_x, klip_params, mtx_x_signal_only
    )
    # ... blink and you'll miss it

    if klip_params.initial_decomposition_only:
        return decomposition

    start_idx = 0
    cubes, signal_arrs, mean_images = [], [], []
    for input_data in klip_inputs:
        # first axis selects "which axis", second axis has an entry per retained pixel
        n_features = np.count_nonzero(input_data.estimation_mask)
        end_idx = start_idx + n_features
        # slice out the range of rows in the combined matrix that correspond to this input
        submatrix = subtracted_mtx[start_idx:end_idx]
        sub_mean_vec = mean_vec[start_idx:end_idx]
        # log.debug(f"{submatrix=}")
        cube = improc.wrap_matrix(
            submatrix,
            input_data.estimation_mask,
            fill_value=klip_params.missing_data_value,
        )
        if signal_mtx is not None:
            signal = improc.wrap_matrix(
                signal_mtx[start_idx:end_idx],
                input_data.estimation_mask,
                fill_value=klip_params.missing_data_value,
            )
        else:
            signal = None

        mean_image = improc.wrap_vector(
            sub_mean_vec,
            input_data.estimation_mask,
            fill_value=klip_params.missing_data_value
        )
        cubes.append(cube)
        mean_images.append(mean_image)
        signal_arrs.append(signal)
        start_idx += n_features

    return cubes, mean_images, signal_arrs


def klip_one(klip_input: KlipInput, klip_params: KlipParams):
    result = klip_multi([klip_input], klip_params)
    if klip_params.initial_decomposition_only:
        return result
    else:
        cubes, means, signals = result
        return cubes[0], means[0], signals[0]


def klip_vapp_separately(
    left_input : KlipInput,
    right_input: KlipInput,
    klip_params : KlipParams,
    vapp_symmetry_angle : float,
    left_over_right_ratios: float
):
    left_result = klip_one(left_input, klip_params)
    right_result = klip_one(right_input, klip_params)
    if klip_params.initial_decomposition_only:
        return left_result, right_result
    else:
        left_cube, left_mean = left_result
        right_cube, right_mean = right_result
        final_cube = vapp_stitch(left_cube, right_cube, vapp_symmetry_angle, left_over_right_ratios)
        final_mean = vapp_stitch(left_mean[np.newaxis,:], right_mean[np.newaxis,:], vapp_symmetry_angle, left_over_right_ratios)
        return final_cube, final_mean

def vapp_stitch(
    left_cube,
    right_cube,
    vapp_symmetry_angle: float,
    left_over_right_ratios: float=1.0
):
    log.debug("begin vapp_stitch")
    if right_cube.shape != left_cube.shape:
        raise ValueError("Left and right vAPP cubes must be the same shape")
    plane_shape = left_cube.shape[1:]
    left_half, right_half = vapp.mask_along_angle(plane_shape, vapp_symmetry_angle)
    final_cube = improc.combine_paired_cubes(
        left_cube,
        left_over_right_ratios * right_cube, # right * (scale_left / scale_right) = scaled right
        left_half,
        right_half,
    )
    log.debug("vapp_stitch finished")
    return final_cube


def adi(cube: np.ndarray, derotation_angles: np.ndarray, operation : CombineOperation):
    derot_cube = improc.derotate_cube(cube, derotation_angles)
    out_image = improc.combine(derot_cube, operation)
    return out_image

def adi_coverage(combination_mask, derotation_angles):
    coverage_cube = np.repeat(combination_mask[np.newaxis, :, :], len(derotation_angles), axis=0)
    return adi(coverage_cube, derotation_angles, operation=CombineOperation.SUM)

def klip(
    sci_arr: da.core.Array,
    estimation_mask: np.ndarray,
    combination_mask: np.ndarray,
    exclude_nearest_n_frames: int,
    k_klip: int,
):
    log.debug("Assembling pipeline...")
    mtx_x = improc.unwrap_cube(sci_arr, estimation_mask)
    log.debug(f"{mtx_x.shape=}")

    subtracted_mtx = starlight_subtraction.klip_mtx(
        mtx_x, k_klip, exclude_nearest_n_frames
    )
    outcube = improc.wrap_matrix(subtracted_mtx, estimation_mask)
    # TODO apply combination_mask
    log.debug(f"{outcube.shape=}")
    log.debug("done assembling")
    return outcube

def evaluate_starlight_subtraction(
    klip_input: KlipInput,
    derotation_angles: np.ndarray,
    specs: List[characterization.CompanionSpec],
    template_psf: np.ndarray,
    klip_params: KlipParams,
    aperture_diameter_px: float,
    apertures_to_exclude: int,
    adi_combine_by : CombineOperation,
    template_scale_factors: Optional[np.ndarray] = None,
    saturation_threshold: Optional[float] = None
):
    injected_sci_arr, _ = characterization.inject_signals(
        klip_input.sci_arr, specs, template_psf,
        angles=derotation_angles,
        template_scale_factors=template_scale_factors,
        saturation_threshold=saturation_threshold
    )
    klip_input.sci_arr = injected_sci_arr
    outcube, mean_image, signals = klip_one(
        klip_input,
        klip_params,
    )
    out_image = adi(outcube, derotation_angles, adi_combine_by)

    recovered_signals = characterization.recover_signals(
        out_image, specs, aperture_diameter_px, apertures_to_exclude
    )
    return recovered_signals
