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
from ..core import PipelineCollection

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
    exclude: Union[List[improc.BBox], None],
    ext=0,
    dq_ext='DQ'
):
    coll = input_coll.map(
        sky_model.background_subtract,
        model_sky,
        mask_dilate_iters,
        n_sigma=n_sigma,
        exclude=exclude,
        ext=ext,
        dq_ext=dq_ext,
    )
    return coll


def align_to_templates(
    input_coll: PipelineCollection,
    cutout_specs: List[improc.CutoutTemplateSpec],
    upsample_factor: int = 100,
    ext: Union[int, str] = 0,
    dq_ext: Union[int, str] = "DQ",
) -> PipelineCollection:
    log.debug(f'align_to_templates {cutout_specs=}')
    # explode list of cutout_specs into individual cutout pipelines
    d_hdus_for_cutouts = []
    for cspec in cutout_specs:
        d_hdus = (input_coll
                  .map(data_quality.get_masked_data, ext=ext, dq_ext=dq_ext)
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
    dq_ext: Union[int, str] = 'DQ'
):
    log.debug("Assembling compute_sky_model pipeline...")
    sky_cube = iofits.hdulists_to_dask_cube(inputs_collection.items, plane_shape, ext=ext)
    dq_cube = iofits.hdulists_to_dask_cube(inputs_collection.items, plane_shape, ext=dq_ext, dtype=int)
    badpix_arr = np.bitwise_or.reduce(dq_cube, axis=0)
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


def klip_multi(klip_inputs: List[KlipInput], klip_params: KlipParams):
    log.debug("assembling klip_multi")
    matrices = []
    subset_indices = []
    xp = core.get_array_module(klip_inputs[0].sci_arr)
    for idx, input_data in enumerate(klip_inputs):
        mtx_x, subset_idxs = improc.unwrap_cube(
            input_data.sci_arr, input_data.estimation_mask
        )
        log.debug(
            f"klip input {idx} has {mtx_x.shape=} from {input_data.sci_arr.shape=} and {np.count_nonzero(input_data.estimation_mask)=} giving {subset_idxs.shape=}"
        )
        matrices.append(mtx_x)
        subset_indices.append(subset_idxs)

    mtx_x = xp.vstack(matrices)
    subtracted_mtx, mean_vec = starlight_subtraction.klip_mtx(
        mtx_x, klip_params
    )
    start_idx = 0
    cubes, mean_images = [], []
    for input_data, subset_idxs in zip(klip_inputs, subset_indices):
        # first axis selects "which axis", second axis has an entry per retained pixel
        n_features = subset_idxs.shape[1]
        end_idx = start_idx + n_features
        if input_data.combination_mask is not None:
            # slice out the range of rows in the combined matrix that correspond to this input
            submatrix = subtracted_mtx[start_idx:end_idx]
            sub_mean_vec = mean_vec[start_idx:end_idx]
            # log.debug(f"{submatrix=}")
            cube = improc.wrap_matrix(
                submatrix,
                input_data.sci_arr.shape,
                subset_idxs,
                fill_value=klip_params.missing_data_value,
            )
            mean_image = improc.wrap_vector(
                sub_mean_vec,
                input_data.sci_arr.shape[1:],
                subset_idxs,
                fill_value=klip_params.missing_data_value
            )
            # log.debug(
            #     f"after slicing submatrix and rewrapping with indices from estimation mask: {cube=}"
            # )
            # TODO is there a better way?
            re_unwrapped_cube, subset_idxs = improc.unwrap_cube(
                cube, input_data.combination_mask
            )
            re_unwrapped_mean_image, im_idxs = improc.unwrap_image(mean_image, input_data.combination_mask)
            # log.debug(f"{unwrapped=} {subset_idxs.shape=}")
            cube = improc.wrap_matrix(
                re_unwrapped_cube,
                cube.shape,
                subset_idxs,
                fill_value=klip_params.missing_data_value,
            )
            mean_image = improc.wrap_vector(
                re_unwrapped_mean_image,
                cube.shape[1:],
                subset_idxs,
                fill_value=klip_params.missing_data_value,
            )
            # log.debug(f"after rewrapping with combination mask indices: {cube=}")
        else:
            cube = None
        cubes.append(cube)
        mean_images.append(mean_image)
        start_idx += n_features

    return cubes, mean_images


def klip_one(klip_input: KlipInput, klip_params: KlipParams):
    cubes, means = klip_multi([klip_input], klip_params)
    return cubes[0], means[0]


def klip_vapp_separately(left_input : KlipInput, right_input: KlipInput, klip_params : KlipParams, vapp_symmetry_angle : float, left_over_right_ratios: float):
    left_cube, left_mean = klip_one(left_input, klip_params)
    right_cube, right_mean = klip_one(right_input, klip_params)

    final_cube = vapp_stitch(left_cube, right_cube, vapp_symmetry_angle, left_over_right_ratios)
    final_mean = vapp_stitch(left_mean[np.newaxis,:], right_mean[np.newaxis,:], vapp_symmetry_angle)
    return final_cube, final_mean

def vapp_stitch(
    left_cube,
    right_cube,
    vapp_symmetry_angle: float,
    left_over_right_ratios: float
):
    log.debug("begin vapp_stitch")
    if right_cube.shape != left_cube.shape:
        raise ValueError("Left and right vAPP cubes must be the same shape")
    plane_shape = left_cube.shape[1:]
    left_half, right_half = vapp.mask_along_angle(plane_shape, vapp_symmetry_angle)
    final_cube = improc.combine_paired_cubes(
        left_cube,
        left_over_right_ratios * right_cube,
        left_half,
        right_half,
    )
    log.debug("vapp_stitch finished")
    return final_cube


def adi(cube: np.ndarray, derotation_angles: np.ndarray, operation : CombineOperation):
    derot_cube = improc.derotate_cube(cube, derotation_angles)
    out_image = improc.combine_cube(derot_cube, operation)
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
    mtx_x, subset_idxs = improc.unwrap_cube(sci_arr, estimation_mask)
    log.debug(f"{mtx_x.shape=}")

    subtracted_mtx = starlight_subtraction.klip_mtx(
        mtx_x, k_klip, exclude_nearest_n_frames
    )
    outcube = improc.wrap_matrix(subtracted_mtx, sci_arr.shape, subset_idxs)
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
    outcube, mean_image = klip_one(
        KlipInput(
            injected_sci_arr, klip_input.estimation_mask, klip_input.combination_mask
        ),
        klip_params,
    )
    out_image = adi(outcube, derotation_angles, adi_combine_by)

    recovered_signals = characterization.recover_signals(
        out_image, specs, aperture_diameter_px, apertures_to_exclude
    )
    return recovered_signals
