from importlib import resources
import dask
import numpy as np
import pytest

from .. import pipelines
from .. import core, constants

from . import starlight_subtraction, improc, characterization, learning

ABSIL_GOOD_SNR_THRESHOLD = 8.36


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
def test_end_to_end(xp, decomposer, snr_threshold):
    res_handle = resources.open_binary(
        "xpipeline.ref", "naco_betapic_preproc_absil2013_gonzalez2017.npz"
    )
    data = np.load(res_handle)
    n_modes = 50
    threshold = 2200  # fake, just to test masking
    good_pix_mask = xp.asarray(np.average(data["cube"], axis=0) < threshold)
    cube = xp.asarray(data["cube"])
    image_vecs, subset_idxs = improc.unwrap_cube(cube, good_pix_mask)
    starlight_subtracted = starlight_subtraction.klip_to_modes(
        image_vecs,
        decomposer,
        n_modes,
    )

    outcube = improc.wrap_matrix(starlight_subtracted, cube.shape, subset_idxs)
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
    (constants.KlipStrategy.COVARIANCE, learning.eigh_top_k)])
def test_klip_mtx(strategy, decomposer):
    res_handle = resources.open_binary(
        "xpipeline.ref", "naco_betapic_preproc_absil2013_gonzalez2017.npz"
    )
    data = np.load(res_handle)
    n_modes = 50
    threshold = 2200  # fake, just to test masking
    good_pix_mask = np.average(data["cube"], axis=0) < threshold
    cube = data["cube"]
    image_vecs, subset_idxs = improc.unwrap_cube(cube, good_pix_mask)

    params = starlight_subtraction.KlipParams(
        k_klip=n_modes,
        exclusions=[],
        decomposer=decomposer,
        reuse=False,
        initial_decomposer=None,
        strategy=strategy,
    )
    starlight_subtracted = starlight_subtraction.klip_mtx(image_vecs, params)

    outcube = improc.wrap_matrix(starlight_subtracted, cube.shape, subset_idxs)
    final_cube = improc.derotate_cube(outcube, data["angles"])
    final_image = np.nansum(final_cube, axis=0)

    r_px, pa_deg = 18.4, -42.8
    fwhm_naco = 4

    locations, results = characterization.reduce_apertures(
        final_image, r_px, pa_deg, fwhm_naco, np.sum
    )
    snr = characterization.calc_snr_mawet(results[0], results[1:])
    assert snr > ABSIL_GOOD_SNR_THRESHOLD
