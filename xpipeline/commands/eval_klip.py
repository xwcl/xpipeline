import xconf
import logging
from typing import Optional

# from . import constants as const
from .. import utils
# from .. import pipelines  # , irods
# from ..core import LazyPipelineCollection
# from ..tasks import iofits, characterization

from .klip import Klip

log = logging.getLogger(__name__)

@xconf.config
class SearchConfig:
    min_r_px : float = xconf.field(default=None, help="Limit blind search to pixels more than this radius from center")
    max_r_px : float = xconf.field(default=None, help="Limit blind search to pixels less than this radius from center")
    snr_threshold : float = xconf.field(default=5.0, help="Threshold above which peaks of interest should be reported")


@xconf.config
class CompanionConfig:
    scale : float = xconf.field(help=utils.unwrap(
        """Scale factor multiplied by template to give companion,
        i.e., contrast ratio. Can be negative or zero."""))
    r_px : float = xconf.field(help="Radius at which to measure SNR")
    pa_deg : float = xconf.field(help="Position angle in degrees East of North at which to measure SNR")


@xconf.config
class TemplateConfig:
    path : str = xconf.field(help=utils.unwrap(
        """Path to FITS image of template PSF, scaled to the
        average amplitude of the host star signal such that
        multiplying by the contrast gives an appropriately
        scaled planet PSF"""
    ))
    scale_factors_path : str = xconf.field(help=utils.unwrap(
        """Path to FITS file with extensions for each data extension
        containing 1D arrays of scale factors that match template PSF
        intensity to real PSF intensity per-frame"""
    ))


@xconf.config
class EvalKlip(Klip):
    "Inject and recover a companion in ADI data through KLIP"
    template : TemplateConfig = xconf.field(help="Paths for template image and scale factors")
    companions : Optional[list[CompanionConfig]] = xconf.field(help="Companions to inject (optionally) and measure SNR for")
    aperture_diameter_px : float = xconf.field(help="Diameter of the SNR estimation aperture (~lambda/D) in pixels")
    apertures_to_exclude : int = xconf.field(default=1, help=utils.unwrap(
        """Number of apertures on *each side* of the specified target
        location to exclude when calculating the noise (in other
        words, a value of 1 excludes two apertures total,
        one on each side)"""))
    search : SearchConfig = xconf.field(help="Configure blind search", default=SearchConfig())
    output_klip_final : bool = xconf.field(default=True, help="Whether to save the final KLIPped image for inspection")
    output_injected_data : bool = xconf.field(default=False, help="Whether to save the data with fake signals for inspection")

    def __post_init__(self):
        if self.companions is None:
            self.companions = []
        return super().__post_init__()

    def _load_inputs(self, *args, **kwargs):
        from ..tasks import iofits
        sci_arr, rot_arr, region_mask = super()._load_inputs(*args, **kwargs)
        template_psf = iofits.load_fits_from_path(self.template.path)[0].data
        return sci_arr, rot_arr, region_mask, template_psf

    def _load_companions(self):
        from ..tasks import characterization
        specs = []
        for companion in self.companions:
            specs.append(characterization.CompanionSpec(
                scale=companion.scale,
                r_px=companion.r_px,
                pa_deg=companion.pa_deg
            ))
        return specs

    def main(self):
        import fsspec.spec
        from pprint import pformat
        import orjson
        import dataclasses
        import numpy as np
        import dask
        import time
        from ..tasks import iofits, characterization
        from .. import pipelines
        start = time.perf_counter()
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        dest_fs.makedirs(destination, exist_ok=True)
        output_result_fn = utils.join(self.destination, "result.json")
        output_klip_final_fn = utils.join(self.destination, "klip_final.fits")
        output_mean_image_fn = utils.join(self.destination, "mean_image.fits")
        output_coverage_map_fn = utils.join(self.destination, "coverage_map.fits")
        output_injected_data_fn = utils.join(self.destination, "injected_data.fits")
        outputs = [output_result_fn]
        if self.output_klip_final:
            outputs.append(output_klip_final_fn)
        if self.output_mean_image:
            outputs.append(output_mean_image_fn)
        if self.output_coverage_map:
            outputs.append(output_coverage_map_fn)
        if self.output_injected_data:
            outputs.append(output_injected_data_fn)
        self.quit_if_outputs_exist(outputs)

        specs = self._load_companions()
        aperture_diameter_px = self.aperture_diameter_px
        apertures_to_exclude = self.apertures_to_exclude
        template_hdul = iofits.load_fits_from_path(self.template.path)

        dataset_path = self.input
        template_scale_factors = iofits.load_fits_from_path(self.template.scale_factors_path)

        # process like the klip command
        klip_inputs, obs_method, derotation_angles = self._assemble_klip_inputs(dataset_path)
        klip_params = self._assemble_klip_params(klip_inputs, derotation_angles)

        log.info("Injecting signals")
        if "vapp" in obs_method:
            left_extname = obs_method["vapp"]["left"]
            right_extname = obs_method["vapp"]["right"]
            if left_extname not in template_hdul or right_extname not in template_hdul:
                raise RuntimeError(
                    f"Couldn't find matching template PSFs for extensions named {left_extname} and {right_extname}"
                )
            log.debug(f"vAPP {left_extname=} {right_extname=}")
            klip_inputs[0].sci_arr = characterization.inject_signals(
                klip_inputs[0].sci_arr,
                derotation_angles,
                specs,
                template_hdul[left_extname].data,
                template_scale_factors[left_extname].data,
                saturation_threshold=self.saturation_threshold
            )
            log.debug("Injected left")
            klip_inputs[1].sci_arr = characterization.inject_signals(
                klip_inputs[1].sci_arr,
                derotation_angles,
                specs,
                template_hdul[left_extname].data,
                template_scale_factors[left_extname].data,
                saturation_threshold=self.saturation_threshold
            )
            log.debug("Injected right")
            if self.output_injected_data:
                injected_data_hdul = iofits.DaskHDUList([
                    iofits.DaskHDU(klip_inputs[0].sci_arr, name=left_extname),
                    iofits.DaskHDU(klip_inputs[1].sci_arr, name=right_extname),
                    iofits.DaskHDU(derotation_angles, name="ANGLES")
                ])
                iofits.write_fits(injected_data_hdul, output_injected_data_fn)
        else:
            if "SCI" not in template_hdul and len(template_hdul[0].data.shape) == 0:
                raise RuntimeError(
                    f"No 'SCI' extension in {self.template_path} and no data in primary extension"
                )
            if "SCI" in template_hdul:
                ext = "SCI"
            else:
                ext = 0
            template_psf = template_hdul[ext].data
            klip_inputs[0].sci_arr = characterization.inject_signals(
                klip_inputs[0].sci_arr, derotation_angles, specs, template_psf, template_scale_factors[ext].data,
                saturation_threshold=self.saturation_threshold
            )

        import time
        start = time.perf_counter()
        outcubes, outmeans = self._klip(klip_inputs, klip_params, obs_method)
        out_image, mean_image, coverage_image = self._assemble_out_images(klip_inputs, obs_method, outcubes, outmeans, derotation_angles)
        elapsed = time.perf_counter() - start
        log.info(f"Computed in {elapsed} sec")


        recovered_signals = characterization.recover_signals(
            out_image, specs, aperture_diameter_px, apertures_to_exclude
        )
        if self.search.iwa_px is None:
            self.search.iwa_px = self.mask_iwa_px
        if self.search.owa_px is None:
            self.search.owa_px = self.mask_owa_px
        all_candidates, (iwa_px, owa_px) = characterization.locate_snr_peaks(
            out_image, aperture_diameter_px, self.search.min_r_px, self.search.max_r_px, apertures_to_exclude, self.search.snr_threshold
        )

        if self.output_klip_final:
            iofits.write_fits(
                iofits.DaskHDUList([iofits.DaskHDU(out_image)]), output_klip_final_fn
            )
        if self.output_mean_image:
            iofits.write_fits(
                iofits.DaskHDUList([iofits.DaskHDU(mean_image)]), output_mean_image_fn
            )
        if self.output_coverage_map:
            iofits.write_fits(
                iofits.DaskHDUList([iofits.DaskHDU(coverage_image)]), output_coverage_map_fn
            )

        end = time.perf_counter()
        time_elapsed_sec = end - start
        log.info(f"Done in {time_elapsed_sec} sec")
        payload = xconf.asdict(self)
        payload['recovered_signals'] = [dataclasses.asdict(x) for x in recovered_signals]
        payload['candidates'] = [dataclasses.asdict(x) for x in all_candidates]
        payload['time_elapsed_sec'] = time_elapsed_sec
        log.info(f"Result of KLIP + ADI signal injection and recovery:")
        log.info(pformat(payload))

        with fsspec.open(output_result_fn, "wb") as fh:
            payload_str = orjson.dumps(
                payload, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
            )
            fh.write(payload_str)
            fh.write(b"\n")

        return output_result_fn
