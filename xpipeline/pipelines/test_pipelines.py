import pytest
from .new import (
    StarlightSubtractPipeline,
    StarlightSubtractionDataConfig,
    KlipTransposePipeline,
    PipelineInputConfig,
    ModelSignalInputConfig,
    RadialMaskConfig,
    CompanionConfig,
    PreloadedArray,
    KlipTranspose,
    Klip,
    KModesValuesConfig,
)

from ..tasks.characterization import calculate_snr
from ..tasks.test_common import naco_betapic_data
import numpy as np

@pytest.mark.parametrize(
    'strategy_cls, snr_threshold',
    [(KlipTranspose, 31.24), (Klip, 15.63)]
)
def test_klip_pipeline(naco_betapic_data, strategy_cls, snr_threshold):
    threshold = 2200  # fake, just to test masking
    cube = naco_betapic_data["cube"]
    good_pix_mask = np.average(cube, axis=0) < threshold
    psf = naco_betapic_data["psf"] / np.max(naco_betapic_data["psf"])
    scale_factors = np.max(cube, axis=(1, 2))
    angles = naco_betapic_data["angles"]
    input_config = PipelineInputConfig(
        sci_arr=PreloadedArray(cube),
        estimation_mask=PreloadedArray(good_pix_mask),
        combination_mask=PreloadedArray(good_pix_mask),
        radial_mask=RadialMaskConfig(min_r_px=0, max_r_px=40),
        model_inputs=ModelSignalInputConfig(
            model=PreloadedArray(psf),
            scale_factors=PreloadedArray(scale_factors),
        )
    )
    r_px, pa_deg = 18.4, -42.8
    data_config = StarlightSubtractionDataConfig(
        inputs=[input_config],
        angles=PreloadedArray(angles),
        companion=CompanionConfig(
            r_px=r_px,
            pa_deg=pa_deg,
            scale=0.0,
        ),
    )
    aperture_diameter_px = 4
    k_modes_values = [5]
    pl = StarlightSubtractPipeline(
        data=data_config,
        strategy=strategy_cls(),
        k_modes=KModesValuesConfig(values=k_modes_values),
    )
    result = pl.execute()
    finim = result.modes[k_modes_values[0]].destination_images["finim"]
    snr = calculate_snr(
        finim,
        r_px,
        pa_deg,
        aperture_diameter_px,
        exclude_nearest=1,
    )
    assert snr_threshold < snr
    # data = naco_betapic_data

    # threshold = 2200  # fake, just to test masking
    # good_pix_mask = np.average(data["cube"], axis=0) < threshold
    # cube = data["cube"]
    # image_vecs = improc.unwrap_cube(cube, good_pix_mask)

    # r_px, pa_deg = 18.4, -42.8
    # psf = data["psf"]
    # psf /= np.max(psf)
    # scale_factors = np.max(cube, axis=(1, 2))
    # angles = data["angles"]
    # spec = characterization.CompanionSpec(r_px=r_px, pa_deg=pa_deg, scale=1.0)
    # _, signal_only = characterization.inject_signals(cube, [spec], psf, angles, scale_factors)
    # model_vecs = improc.unwrap_cube(signal_only, good_pix_mask)
    # # compute basis only first
    # params = starlight_subtraction.TrapParams(k_modes=17, return_basis=True)
    # basis = starlight_subtraction.trap_mtx(image_vecs, model_vecs, params)
    # # pass it back in for real
    # params.precomputed_temporal_basis = basis
    # params.return_basis = False
    # coeff, timers, pix_used, resid_vecs = starlight_subtraction.trap_mtx(image_vecs, model_vecs, params)
    # print(coeff)
    # _ref_value = 0.009980708465446193
    # assert np.abs(coeff - _ref_value) < 1e-4 * _ref_value, "model coeff did not match value when test was written to prevent regressions"

