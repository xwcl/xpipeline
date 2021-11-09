from importlib import resources
import dask
import numpy as np
import pytest

from .. import pipelines
from .. import core, constants

from . import starlight_subtraction, improc, characterization, learning

ABSIL_GOOD_SNR_THRESHOLD = 8.36

@pytest.fixture
def naco_betapic_data():
    res_handle = resources.open_binary(
        "xpipeline.ref", "naco_betapic_preproc_absil2013_gonzalez2017.npz"
    )
    data = np.load(res_handle)
    return data

@pytest.mark.parametrize(
    "xp,decomposer,snr_threshold",
    [
        (
            np,
            starlight_subtraction.MinimalDowndateSVDDecomposer,
            ABSIL_GOOD_SNR_THRESHOLD,
        )
    ],
)
def test_downdate_end_to_end(xp, decomposer, snr_threshold, naco_betapic_data):
    data = naco_betapic_data
    n_modes = 50
    threshold = 2200  # fake, just to test masking
    good_pix_mask = xp.asarray(np.average(data["cube"], axis=0) < threshold)
    cube = xp.asarray(data["cube"])
    image_vecs = improc.unwrap_cube(cube, good_pix_mask)
    starlight_subtracted = starlight_subtraction.klip_to_modes(
        image_vecs,
        decomposer,
        n_modes,
    )

    outcube = improc.wrap_matrix(starlight_subtracted, good_pix_mask)
    final_cube = improc.derotate_cube(outcube, data["angles"])
    final_image = np.nansum(final_cube, axis=0)

    r_px, pa_deg = 18.4, -42.8
    fwhm_naco = 4

    locations, results = characterization.reduce_apertures(
        final_image, r_px, pa_deg, fwhm_naco, np.sum
    )
    snr = characterization.calc_snr_mawet(results[0], results[1:])
    assert snr > snr_threshold

@pytest.mark.parametrize(
    "strategy, decomposer",
    [(constants.KlipStrategy.DOWNDATE_SVD, learning.generic_svd),
    (constants.KlipStrategy.SVD, learning.generic_svd),
    (constants.KlipStrategy.COVARIANCE, learning.eigh_top_k),
    (constants.KlipStrategy.COVARIANCE, learning.eigh_full_decomposition)])
def test_klip_mtx(strategy, decomposer, naco_betapic_data):
    data = naco_betapic_data
    n_modes = 50
    threshold = 2200  # fake, just to test masking
    good_pix_mask = np.average(data["cube"], axis=0) < threshold
    cube = data["cube"]
    image_vecs = improc.unwrap_cube(cube, good_pix_mask)

    params = starlight_subtraction.KlipParams(
        k_klip=n_modes,
        exclusions=[],
        decomposer=decomposer,
        reuse=False,
        initial_decomposer=None,
        strategy=strategy,
    )
    (starlight_subtracted, _), mean_vec = starlight_subtraction.klip_mtx(image_vecs, params)

    outcube = improc.wrap_matrix(starlight_subtracted, good_pix_mask)
    final_cube = improc.derotate_cube(outcube, data["angles"])
    final_image = np.nansum(final_cube, axis=0)

    r_px, pa_deg = 18.4, -42.8
    fwhm_naco = 4

    locations, results = characterization.reduce_apertures(
        final_image, r_px, pa_deg, fwhm_naco, np.sum
    )
    snr = characterization.calc_snr_mawet(results[0], results[1:])
    assert snr > ABSIL_GOOD_SNR_THRESHOLD


def test_trap_mtx(naco_betapic_data):
    data = naco_betapic_data

    threshold = 2200  # fake, just to test masking
    good_pix_mask = np.average(data["cube"], axis=0) < threshold
    cube = data["cube"]
    image_vecs = improc.unwrap_cube(cube, good_pix_mask)

    params = starlight_subtraction.TrapParams(k_modes=17)
    r_px, pa_deg = 18.4, -42.8
    psf = data["psf"]
    psf /= np.max(psf)
    scale_factors = np.max(cube, axis=(1, 2))
    angles = data["angles"]
    spec = characterization.CompanionSpec(r_px=r_px, pa_deg=pa_deg, scale=1.0)
    signal_only = characterization.generate_signals(cube.shape, [spec], psf, angles, scale_factors)
    model_vecs = improc.unwrap_cube(signal_only, good_pix_mask)
    coeff, timers, pix_used, resid_vecs = starlight_subtraction.trap_mtx(image_vecs, model_vecs, params)

    outcube = improc.wrap_matrix(resid_vecs, good_pix_mask)
    final_cube = improc.derotate_cube(outcube, angles)
    final_image = np.nansum(final_cube, axis=0)

    fwhm_naco = 4

    _, results = characterization.reduce_apertures(
        final_image, r_px, pa_deg, fwhm_naco, np.sum
    )
    snr = characterization.calc_snr_mawet(results[0], results[1:])
    assert snr > 35, "snr did not meet threshold based on performance when test was written to prevent regressions"

    contrast = -0.02  # not real, just empirically what cancels the planet signal
    image_vecs_2 = improc.unwrap_cube(cube + contrast * signal_only, good_pix_mask)
    coeff_2, timers_2, pix_used_2, resid_vecs_2 = starlight_subtraction.trap_mtx(image_vecs_2, model_vecs, params)

    outcube_2 = improc.wrap_matrix(resid_vecs_2, good_pix_mask)
    final_cube_2 = improc.derotate_cube(outcube_2, angles)
    final_image_2 = np.nansum(final_cube_2, axis=0)

    _, results_2 = characterization.reduce_apertures(
        final_image_2, r_px, pa_deg, fwhm_naco, np.sum
    )
    snr = characterization.calc_snr_mawet(results_2[0], results_2[1:])
    assert snr < 1, "snr for signal-free cube too high"



def test_trap_mtx_reuse_basis(naco_betapic_data):
    data = naco_betapic_data

    threshold = 2200  # fake, just to test masking
    good_pix_mask = np.average(data["cube"], axis=0) < threshold
    cube = data["cube"]
    image_vecs = improc.unwrap_cube(cube, good_pix_mask)

    r_px, pa_deg = 18.4, -42.8
    psf = data["psf"]
    psf /= np.max(psf)
    scale_factors = np.max(cube, axis=(1, 2))
    angles = data["angles"]
    spec = characterization.CompanionSpec(r_px=r_px, pa_deg=pa_deg, scale=1.0)
    _, signal_only = characterization.inject_signals(cube, [spec], psf, angles, scale_factors)
    model_vecs = improc.unwrap_cube(signal_only, good_pix_mask)
    # compute basis only first
    params = starlight_subtraction.TrapParams(k_modes=17, return_basis=True)
    basis = starlight_subtraction.trap_mtx(image_vecs, model_vecs, params)
    # pass it back in for real
    params.precomputed_temporal_basis = basis
    params.return_basis = False
    coeff, timers, pix_used, resid_vecs = starlight_subtraction.trap_mtx(image_vecs, model_vecs, params)
    print(coeff)
    _ref_value = 0.009980708465446193
    assert np.abs(coeff - _ref_value) < 1e-4 * _ref_value, "model coeff did not match value when test was written to prevent regressions"

def test_trap_detection_mapping(naco_betapic_data):
    cube = naco_betapic_data['cube']
    avg_amp = np.average(np.sum(cube, axis=0))
    scaled_psf = naco_betapic_data['psf'] / np.sum(naco_betapic_data['psf']) * avg_amp
    N_POINTS = 8
    rho, _ = improc.polar_coords(improc.arr_center(cube[0]), cube[0].shape)
    iwa_px = 14 # just inside beta pic b in these data
    good_pix_mask = rho > iwa_px
    r_px = iwa_px + 5
    pa_degs = (360 / N_POINTS) * np.arange(N_POINTS)
    image_vecs = improc.unwrap_cube(cube, good_pix_mask)
    trap_params = starlight_subtraction.TrapParams(k_modes=3)
    xx = []
    yy = []
    coeffs = []
    for i in range(N_POINTS):
        model_cube = characterization.generate_signals(
            cube.shape,
            [characterization.CompanionSpec(r_px=r_px, pa_deg=pa_degs[i], scale=1)],
            scaled_psf,
            naco_betapic_data['angles']
        )
        model_vecs = improc.unwrap_cube(model_cube, good_pix_mask)
        model_coeff, timers, pix_used, maybe_resid_vecs = starlight_subtraction.trap_mtx(image_vecs, model_vecs, trap_params)
        coeffs.append(model_coeff)
        x, y = characterization.r_pa_to_x_y(r_px, pa_degs[i], 0, 0)
        xx.append(x)
        yy.append(y)
    coeffs = np.asarray(coeffs)
    sigma = characterization.sigma_mad(coeffs)
    assert np.count_nonzero(coeffs / sigma >= 5) == 1, "Exactly one point should be beta Pic b"
