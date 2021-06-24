import pytest
from importlib import resources
import numpy as np
import dask
from .characterization import (
    simple_aperture_locations,
    inject_signals,
    CompanionSpec,
    reduce_apertures,
    calc_snr_mawet,
)
from . import improc
from .. import core, pipelines, constants
from xpipeline.tasks import characterization, starlight_subtraction

da = core.dask_array


def test_simple_aperture_locations():
    r_px = 5
    pa_deg = 0
    diam = 7
    assert np.allclose(
        np.asarray(list(simple_aperture_locations(r_px, pa_deg, diam))),
        [[0, 5], [-5, 0], [0, -5], [5, 0]],
    )
    assert np.allclose(
        np.asarray(
            list(simple_aperture_locations(r_px, pa_deg, diam, exclude_planet=True))
        ),
        [[-5, 0], [0, -5], [5, 0]],
    )
    assert np.allclose(
        np.asarray(
            list(
                simple_aperture_locations(
                    r_px, pa_deg, diam, exclude_planet=True, exclude_nearest=1
                )
            )
        ),
        [[0, -5]],
    )


def test_inject_signals():
    template = np.zeros((4, 4))
    base_pix_val = 10
    template[1:3, 1:3] = base_pix_val
    npix = 128
    n_frames = 4
    angles = np.array([0, 90, 180, 270])
    f_frame = improc.f_test(npix)
    from skimage.transform import rotate

    # n.b. skimage rotate() uses theta in the opposite sense from
    # improc.quick_derotate, but that's actually what we want here
    # since we're faking data that will then be derotated
    cube = np.asarray([rotate(f_frame, theta) for theta in angles])
    r_px = 40
    theta_deg = 90
    scale_val = 5.1234
    specs = [CompanionSpec(scale=scale_val, r_px=r_px, pa_deg=theta_deg)]

    out_pix_val = base_pix_val * scale_val

    # NumPy only
    outcube = inject_signals(cube, angles, specs, template)
    assert np.isclose(outcube[0][128 // 2, 128 // 2 - r_px], out_pix_val)
    assert np.isclose(outcube[1][128 // 2 + r_px, 128 // 2], out_pix_val)
    assert np.isclose(outcube[2][128 // 2, 128 // 2 + r_px], out_pix_val)
    assert np.isclose(outcube[3][128 // 2 - r_px, 128 // 2], out_pix_val)

    # With Dask
    d_cube = da.from_array(cube).rechunk((n_frames, -1, -1))
    d_angles = da.from_array(angles).rechunk((25,))  # provoke mismatch in dimensions
    outcube = inject_signals(d_cube, d_angles, specs, template)
    outcube = outcube.compute()
    angles = d_angles.compute()
    assert np.isclose(outcube[0][128 // 2, 128 // 2 - r_px], out_pix_val)
    assert np.isclose(outcube[1][128 // 2 + r_px, 128 // 2], out_pix_val)
    assert np.isclose(outcube[2][128 // 2, 128 // 2 + r_px], out_pix_val)
    assert np.isclose(outcube[3][128 // 2 - r_px, 128 // 2], out_pix_val)


@pytest.mark.parametrize('strategy,reuse,snr_threshold,decomposer', [
    (constants.KlipStrategy.COVARIANCE, False, 21.2, starlight_subtraction._eigh_full_decomposition),
    (constants.KlipStrategy.COVARIANCE, True, 21.2, starlight_subtraction._eigh_full_decomposition),
    (constants.KlipStrategy.COVARIANCE, False, 21.2, starlight_subtraction._eigh_top_k),
    (constants.KlipStrategy.COVARIANCE, True, 21.2, starlight_subtraction._eigh_top_k),
    (constants.KlipStrategy.DOWNDATE_SVD, False, 21.2, None),
])
def test_end_to_end_dask(strategy, reuse, snr_threshold, decomposer):
    res_handle = resources.open_binary(
        "xpipeline.ref", "naco_betapic_preproc_absil2013_gonzalez2017.npz"
    )
    data = np.load(res_handle)
    n_modes = 9
    threshold = 2200  # fake, just to test masking
    good_pix_mask = np.average(data["cube"], axis=0) < threshold
    sci_arr = da.asarray(data["cube"])
    rot_arr = da.asarray(data["angles"])

    pristine_input = starlight_subtraction.KlipInput(sci_arr, good_pix_mask, good_pix_mask)
    exclusions = []
    # indices = np.arange(rot_arr.shape[0])
    # exc = starlight_subtraction.ExclusionValues(
    #     exclude_within_delta=exclude.nearest_n_frames,
    #     values=indices,
    #     num_excluded_max=2 * exclude.nearest_n_frames + 1
    # )
    # exclusions.append(exc)
    klip_params = starlight_subtraction.KlipParams(
        n_modes,
        exclusions,
        decomposer=decomposer,
        strategy=strategy,
        chunks={0: 500, 1: -1}
    )

    d_outcube = pipelines.klip_one(pristine_input, klip_params)
    d_output_image = pipelines.adi(d_outcube, rot_arr)
    output_image = dask.compute(d_output_image)[0]
    r_px, pa_deg = 18.4, -42.8
    fwhm_naco = data["fwhm"]

    _, results = reduce_apertures(
        output_image,
        r_px,
        pa_deg,
        fwhm_naco,
        np.sum,
        exclude_nearest=1,
    )
    snr = calc_snr_mawet(results[0], results[1:])
    assert snr > snr_threshold

    # can we get the same SNR from the image?
    iwa_px, owa_px = 7, 47
    detections = characterization.locate_snr_peaks(output_image, fwhm_naco, iwa_px, owa_px, exclude_nearest=1, snr_threshold=8)
    peak = detections[0]
    # n.b. not the same as the VIP tutorial quotes, but this is here to make sure
    # we don't change locate_snr_peaks outputs by accident
    assert np.isclose(peak.r_px, 16.98528775146303)
    assert np.isclose(peak.pa_deg, 317.3859440303888)
    assert peak.snr > 23.7

    template_psf = data["psf"]
    avg_frame_total = np.average(np.sum(data["cube"], axis=(1, 2)))
    # scale up to star amplitude
    template_psf = template_psf / np.sum(template_psf) * avg_frame_total

    specs = [
        CompanionSpec(r_px=18.4, pa_deg=-42.8, scale=0),
    ]

    d_recovered_signals = pipelines.evaluate_starlight_subtraction(
        pristine_input,
        rot_arr,
        specs,
        template_psf,
        klip_params,
        aperture_diameter_px=data["fwhm"],
        apertures_to_exclude=1,
    )
    recovered_signals = dask.compute(d_recovered_signals)[0]
    assert recovered_signals[0].snr == snr

    # try with an injected signal now
    specs = [CompanionSpec(r_px=30, pa_deg=90, scale=0.001)]
    d_recovered_signals = pipelines.evaluate_starlight_subtraction(
        pristine_input,
        rot_arr,
        specs,
        template_psf,
        klip_params,
        aperture_diameter_px=data["fwhm"],
        apertures_to_exclude=1,
    )
    recovered_signals = dask.compute(d_recovered_signals)[0]
    assert recovered_signals[0].snr > 12.0

def test_calc_snr_mawet():
    noises = [1, -1, 1, -1]  # stddev of 1, mean of zero by construction
    signal = 10
    snr = signal / np.sqrt(1 + 1/len(noises))
    assert np.isclose(characterization.calc_snr_mawet(signal, noises), snr)

def test_r_pa_conversions():
    r_px, pa_deg = characterization.x_y_to_r_pa(10, 0, 0, 0)
    assert np.isclose(r_px, 10)
    assert np.isclose(pa_deg, 270)

    xc, yc = 10, 5
    r_px, pa_deg = characterization.x_y_to_r_pa(10 + xc, 0 + yc, xc, yc)
    assert np.isclose(r_px, 10)
    assert np.isclose(pa_deg, 270)

    x, y = characterization.r_pa_to_x_y(10, 0, 0, 0)
    assert np.isclose(x, 0)
    assert np.isclose(y, 10)

    x, y = characterization.r_pa_to_x_y(10, 0, xc, yc)
    assert np.isclose(x, xc)
    assert np.isclose(y, 10 + yc)

    x, y = characterization.r_pa_to_x_y(10, -90, xc, yc)
    assert np.isclose(x, 10 + xc)
    assert np.isclose(y, yc)

    x, y = characterization.r_pa_to_x_y(10, 270, xc, yc)
    assert np.isclose(x, 10 + xc)
    assert np.isclose(y, yc)
