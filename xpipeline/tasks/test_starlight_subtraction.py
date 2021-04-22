from importlib import resources
import dask
import numpy as np
import pytest

from .. import pipelines
from .. import core
cp = core.cupy
da = core.dask_array
torch = core.torch

from . import starlight_subtraction, improc, characterization, learning

ABSIL_GOOD_SNR_THRESHOLD = 8.36

@pytest.mark.parametrize('xp,decomposer,snr_threshold', [
    (np, starlight_subtraction.MinimalDowndateSVDDecomposer, ABSIL_GOOD_SNR_THRESHOLD)
])
def test_end_to_end(xp, decomposer, snr_threshold):
    res_handle = resources.open_binary('xpipeline.ref', 'naco_betapic_preproc_absil2013_gonzalez2017.npz')
    data = np.load(res_handle)
    n_modes = 50
    threshold = 2200  # fake, just to test masking
    good_pix_mask = xp.asarray(np.average(data['cube'], axis=0) < threshold)
    cube = xp.asarray(data['cube'])
    image_vecs, xx, yy = improc.unwrap_cube(cube, good_pix_mask)
    # image_vecs_meansub, mean_vec = starbgone.mean_subtract_vecs(image_vecs)
    starlight_subtracted = starlight_subtraction.klip_to_modes(
        image_vecs,
        decomposer,
        n_modes,
        # solver=solver
    )
        
    outcube = improc.wrap_matrix(starlight_subtracted, cube.shape, xx, yy)
    final_image = improc.quick_derotate(outcube, data['angles'])

    r_px, pa_deg = 18.4, -42.8
    fwhm_naco = 4

    locations, results = characterization.reduce_apertures(
        final_image,
        r_px,
        pa_deg,
        fwhm_naco,
        np.sum
    )
    snr = characterization.calc_snr_mawet(results[0], results[1:])
    assert snr > snr_threshold


def test_end_to_end_dask():
    res_handle = resources.open_binary('xpipeline.ref', 'naco_betapic_preproc_absil2013_gonzalez2017.npz')
    data = np.load(res_handle)
    n_modes = 9
    threshold = 2200  # fake, just to test masking
    good_pix_mask = np.average(data['cube'], axis=0) < threshold
    sci_arr = da.asarray(data['cube'])
    rot_arr = da.asarray(data['angles'])

    d_output_image = pipelines.klip_adi(
        sci_arr,
        rot_arr,
        good_pix_mask,
        angle_scale=1,
        angle_offset=0,
        exclude_nearest_n_frames=0,
        k_klip_value=n_modes,
    )
    output_image = dask.compute(d_output_image)[0]
    r_px, pa_deg = 18.4, -42.8
    fwhm_naco = data['fwhm']

    _, results = characterization.reduce_apertures(
        output_image,
        r_px,
        pa_deg,
        fwhm_naco,
        np.sum,
        exclude_nearest=1,
    )
    snr = characterization.calc_snr_mawet(results[0], results[1:])
    assert snr > 21.8

    template_psf = data['psf']
    avg_frame_total = np.average(np.sum(data['cube'], axis=(1,2)))
    # scale up to star amplitude
    template_psf = template_psf / np.sum(template_psf) * avg_frame_total
    
    specs = [
        characterization.CompanionSpec(r_px=18.4, pa_deg=-42.8, scale=0),
    ]

    d_recovered_signals = pipelines.evaluate_starlight_subtraction(
        sci_arr,
        rot_arr,
        good_pix_mask,
        specs,
        template_psf,
        angle_scale=1,
        angle_offset=0,
        exclude_nearest_n_frames=0,
        k_klip_value=n_modes,
        aperture_diameter_px=data['fwhm'],
        apertures_to_exclude=1,
    )
    recovered_signals = dask.compute(d_recovered_signals)[0]
    assert recovered_signals[0].snr == snr

    # try with an injected signal now
    specs = [characterization.CompanionSpec(r_px=30, pa_deg=90, scale=0.001)]
    d_recovered_signals = pipelines.evaluate_starlight_subtraction(
        sci_arr,
        rot_arr,
        good_pix_mask,
        specs,
        template_psf,
        angle_scale=1,
        angle_offset=0,
        exclude_nearest_n_frames=0,
        k_klip_value=n_modes,
        aperture_diameter_px=data['fwhm'],
        apertures_to_exclude=1,
    )
    recovered_signals = dask.compute(d_recovered_signals)[0]
    assert recovered_signals[0].snr > 13.5

