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
class CompanionConfig:
    scale : float = xconf.field(help=utils.unwrap(
        """Scale factor multiplied by template to give companion,
        i.e., contrast ratio. Can be negative or zero."""))
    r_px : float = xconf.field(help="Radius at which to measure SNR")
    pa_deg : float = xconf.field(help="Position angle in degrees East of North at which to measure SNR")

@xconf.config
class SearchConfig:
    iwa_px : float = xconf.field(default=None, help="Limit blind search to pixels more than this radius from center")
    owa_px : float = xconf.field(default=None, help="Limit blind search to pixels less than this radius from center")
    snr_threshold : float = xconf.field(default=5.0, help="Threshold above which peaks of interest should be reported")

@xconf.config
class EvalKlip(Klip):
    "Inject and recover a companion in ADI data through KLIP"
    template_path : str = xconf.field(help=utils.unwrap(
        """Path to FITS image of template PSF, scaled to the
        average amplitude of the host star signal such that
        multiplying by the contrast gives an appropriately
        scaled planet PSF"""
    ))
    scale_factors_path : str = xconf.field(help=utils.unwrap(
        """Path to text file with one scale factor per line that matches template intensity to host PSF intensity"""
    ))
    companions : Optional[list[CompanionConfig]] = xconf.field(help="Companions to inject (optionally) and measure SNR for")
    aperture_diameter_px : float = xconf.field(help="Diameter of the SNR estimation aperture (~lambda/D) in pixels")
    apertures_to_exclude : int = xconf.field(default=1, help=utils.unwrap(
        """Number of apertures on *each side* of the specified target
        location to exclude when calculating the noise (in other
        words, a value of 1 excludes two apertures total,
        one on each side)"""))
    search : SearchConfig = xconf.field(help="Configure blind search", default=SearchConfig())

    def __post_init__(self):
        if self.companions is None:
            self.companions = []
        return super().__post_init__()

    def _load_inputs(self, *args, **kwargs):
        from ..tasks import iofits
        sci_arr, rot_arr, region_mask = super()._load_inputs(*args, **kwargs)
        template_psf = iofits.load_fits_from_path(self.template_path)[0].data
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
        from ..tasks import iofits, characterization
        from .. import pipelines
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        dest_fs.makedirs(destination, exist_ok=True)
        output_result = utils.join(self.destination, "result.json")
        output_klip_final = utils.join(self.destination, "eval_klip.fits")
        self.quit_if_outputs_exist([output_result, output_klip_final])

        specs = self._load_companions()
        aperture_diameter_px = self.aperture_diameter_px
        apertures_to_exclude = self.apertures_to_exclude
        template_hdul = iofits.load_fits_from_path(self.template_path)

        dataset_path = self.input
        srcfs = utils.get_fs(dataset_path)

        scale_factors = []
        with srcfs.open(self.scale_factors_path) as fh:
            for line in fh:
                if len(line.strip()) == 0:
                    continue
                value = float(line)
                scale_factors.append(value)
        scale_factors = np.asarray(scale_factors)[::self.sample_every_n]

        # process like the klip command
        klip_inputs, obs_method, derotation_angles = self._assemble_klip_inputs(dataset_path)
        klip_params = self._assemble_klip_params(klip_inputs, derotation_angles)

        # inject signals
        if "vapp" in obs_method:
            left_extname = obs_method["vapp"]["left"]
            right_extname = obs_method["vapp"]["right"]
            if left_extname not in template_hdul or right_extname not in template_hdul:
                raise RuntimeError(
                    f"Couldn't find matching template PSFs for extensions named {left_extname} and {right_extname}"
                )
            klip_inputs[0].sci_arr = characterization.inject_signals(
                klip_inputs[0].sci_arr,
                derotation_angles,
                specs,
                template_hdul[left_extname].data,
                scale_factors,
            )
            klip_inputs[1].sci_arr = characterization.inject_signals(
                klip_inputs[1].sci_arr,
                derotation_angles,
                specs,
                template_hdul[left_extname].data,
                scale_factors,
            )
        else:
            if "SCI" not in template_hdul and len(template_hdul[0].data.shape) == 0:
                raise RuntimeError(
                    f"No 'SCI' extension in {self.template_path} and no data in primary extension"
                )
            if "SCI" in template_hdul:
                template_psf = template_hdul["SCI"].data
            else:
                template_psf = template_hdul[0].data
            klip_inputs[0].sci_arr = characterization.inject_signals(
                klip_inputs[0].sci_arr, derotation_angles, specs, template_psf, scale_factors
            )

        # compose with klip
        outcubes = pipelines.klip_multi(klip_inputs, klip_params)

        # compute final like klip command
        out_image = self._assemble_out_image(obs_method, outcubes, derotation_angles)

        d_recovered_signals = dask.delayed(characterization.recover_signals)(
            out_image, specs, aperture_diameter_px, apertures_to_exclude
        )
        if self.search.iwa_px is None:
            self.search.iwa_px = self.mask_iwa_px
        if self.search.owa_px is None:
            self.search.owa_px = self.mask_owa_px
        self.search.iwa_px, self.search.owa_px = characterization.working_radii_from_aperture_spacing(out_image.shape, self.aperture_diameter_px, apertures_to_exclude, self.search.iwa_px, self.search.owa_px)
        d_all_candidates = dask.delayed(characterization.locate_snr_peaks)(
            out_image, aperture_diameter_px, self.search.iwa_px, self.search.owa_px, apertures_to_exclude, self.search.snr_threshold
        )

        import time
        _ = dask.visualize(out_image, d_recovered_signals, d_all_candidates, filename='eval_klip.svg')
        log.info(f"Computing recovered signals")
        if self.dask.distributed:
            from dask.distributed import performance_report
            with performance_report():
                start = time.perf_counter()
                out_image, recovered_signals, all_candidates = dask.compute(out_image, d_recovered_signals, d_all_candidates)
                end = time.perf_counter()
        else:
            start = time.perf_counter()
            out_image, recovered_signals, all_candidates = dask.compute(out_image, d_recovered_signals, d_all_candidates)
            end = time.perf_counter()
        time_elapsed_sec = end - start
        log.info(f"Done in {time_elapsed_sec} sec")
        iofits.write_fits(
            iofits.DaskHDUList([iofits.DaskHDU(out_image)]), output_klip_final
        )

        payload = xconf.asdict(self)
        payload['recovered_signals'] = [dataclasses.asdict(x) for x in recovered_signals]
        payload['candidates'] = [dataclasses.asdict(x) for x in all_candidates]
        payload['time_elapsed_sec'] = time_elapsed_sec
        log.info(f"Result of KLIP + ADI signal injection and recovery:")
        log.info(pformat(payload))

        with fsspec.open(output_result, "wb") as fh:
            payload_str = orjson.dumps(
                payload, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
            )
            fh.write(payload_str)
            fh.write(b"\n")

        return output_result
