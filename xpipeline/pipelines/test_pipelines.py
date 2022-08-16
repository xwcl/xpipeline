import pytest
from .new import (
    MeasureStarlightSubtractionPipeline,
    StarlightSubtractionMeasurements,
    StarlightSubtractPipeline,
    StarlightSubtract,
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
    PostFilteringResults,
)

from ..tasks.characterization import calculate_snr
from ..tasks.test_common import naco_betapic_data
import numpy as np

STRATEGIES_SNRS = [
    (KlipTranspose, 28), (Klip, 14)
]

@pytest.mark.parametrize(
    'strategy_cls, snr_threshold',
    STRATEGIES_SNRS
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
        times_sec=None,
        angles=PreloadedArray(angles),
        companions=[CompanionConfig(
            r_px=r_px,
            pa_deg=pa_deg,
            scale=0.0,
        )],
    )
    aperture_diameter_px = 4
    k_modes_values = [5]
    pl = StarlightSubtractPipeline(
        data=data_config,
        strategy=strategy_cls(),
        k_modes=KModesValuesConfig(values=k_modes_values),
        resolution_element_px=aperture_diameter_px,
    )
    result = pl.execute()
    pfresult : PostFilteringResults = result.modes[k_modes_values[0]].destination_images["finim"]
    snr = calculate_snr(
        pfresult.unfiltered_image,
        r_px,
        pa_deg,
        aperture_diameter_px,
        exclude_nearest=1,
    )
    assert snr_threshold < snr


@pytest.mark.parametrize(
    'strategy_cls, snr_threshold',
    STRATEGIES_SNRS
)
def test_measure_starlight_subtraction_pipeline(naco_betapic_data, strategy_cls, snr_threshold):
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
        times_sec=None,
        angles=PreloadedArray(angles),
        companions=[CompanionConfig(
            r_px=r_px,
            pa_deg=pa_deg,
            scale=0.0,
        )],
    )
    aperture_diameter_px = 4
    k_modes_values = [5]

    subtraction = StarlightSubtract(
        strategy=strategy_cls(),
        k_modes=KModesValuesConfig(values=k_modes_values),
        resolution_element_px=aperture_diameter_px,
    )
    pl = MeasureStarlightSubtractionPipeline(
        data=data_config,
        subtraction=subtraction,
    )
    result: StarlightSubtractionMeasurements = pl.execute()

    snr = result.by_modes[k_modes_values[0]].by_ext["finim"].tophat.locations[0].snr
    finim = result.by_modes[k_modes_values[0]].by_ext["finim"].unfiltered_image
    recalc_snr = calculate_snr(
        finim,
        r_px,
        pa_deg,
        aperture_diameter_px,
        exclude_nearest=1,
    )
    assert recalc_snr > snr_threshold

def test_klipt_annular_exclusion(naco_betapic_data):
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
        times_sec=None,
        angles=PreloadedArray(angles),
        companions=[CompanionConfig(
            r_px=r_px,
            pa_deg=pa_deg,
            scale=0.0,
        )],
    )
    aperture_diameter_px = 4
    k_modes_values = [5]

    subtraction = StarlightSubtract(
        strategy=KlipTranspose(
            # excluded_annulus_width_px=2 * aperture_diameter_px
        ),
        k_modes=KModesValuesConfig(values=k_modes_values),
        resolution_element_px=aperture_diameter_px,
    )
    pl = MeasureStarlightSubtractionPipeline(
        data=data_config,
        subtraction=subtraction,
    )
    result: StarlightSubtractionMeasurements = pl.execute()
    snr = result.by_modes[k_modes_values[0]].by_ext["finim"].tophat.locations[0].snr
    finim = result.by_modes[k_modes_values[0]].by_ext["finim"].unfiltered_image
    recalc_snr = calculate_snr(
        finim,
        r_px,
        pa_deg,
        aperture_diameter_px,
        exclude_nearest=1,
    )
    print(f"KLIP^T {snr=} {recalc_snr=}")
    assert snr > 27, "True detection SNR too low"
    for i in range(5):
        subtraction.strategy.excluded_annulus_width_px = 8
        data_config.companions[0].pa_deg += 30
        result: StarlightSubtractionMeasurements = pl.execute()
        snr = result.by_modes[k_modes_values[0]].by_ext["finim"].tophat.locations[0].snr
        finim = result.by_modes[k_modes_values[0]].by_ext["finim"].unfiltered_image
        recalc_snr = calculate_snr(
            finim,
            data_config.companions[0].r_px,
            data_config.companions[0].pa_deg,
            aperture_diameter_px,
            exclude_nearest=1,
        )
        print(f"KLIP^T non-detection: {snr=} {recalc_snr=}")
        assert np.abs(snr) < 1, "Spurious detection SNR too high"
