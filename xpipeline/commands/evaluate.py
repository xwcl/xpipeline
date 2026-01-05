import xconf
import logging
from typing import Optional

# from . import constants as const
from .. import utils
# from .. import pipelines  # , irods
# from ..core import LazyPipelineCollection
# from ..tasks import iofits, characterization

from .base import CompanionConfig, BaseCommand, FitsConfig

log = logging.getLogger(__name__)

@xconf.config
class SearchConfig:
    min_r_px : float = xconf.field(default=None, help="Limit blind search to pixels more than this radius from center")
    max_r_px : float = xconf.field(default=None, help="Limit blind search to pixels less than this radius from center")
    snr_threshold : float = xconf.field(default=5.0, help="Threshold above which peaks of interest should be reported")

@xconf.config
class Evaluate(BaseCommand):
    "Evaluate starlight subtracted image for signal detection and/or recovery"
    image : FitsConfig = xconf.field(help="Image to analyze")
    companions : Optional[list[CompanionConfig]] = xconf.field(help="Locations and scales of injected/anticipated companion signals")
    aperture_diameter_px : float = xconf.field(help="Diameter of the SNR estimation aperture (~lambda/D) in pixels")
    apertures_to_exclude : int = xconf.field(default=1, help=utils.unwrap(
        """Number of apertures on *each side* of the specified target
        location to exclude when calculating the noise (in other
        words, a value of 1 excludes two apertures total,
        one on each side)"""))
    search : Optional[SearchConfig] = xconf.field(default=None, help="Configure blind search")
    save_snr_map : bool = xconf.field(default=False, help="Whether to save the SNR map as a FITS image")

    def _load_companions(self):
        import numpy as np
        from ..tasks import characterization
        specs = []
        if self.companions is not None:
            for cconfig in self.companions:
                specs.append(characterization.CompanionSpec(scale=cconfig.scale, r_px=cconfig.r_px, pa_deg=cconfig.pa_deg))
            log.debug(f"Measuring at locations {specs} specified in config")
        return specs


    def main(self):
        import fsspec.spec
        import orjson
        import dataclasses
        import time
        from ..tasks import characterization, iofits
        from .. import pipelines
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        dest_fs.makedirs(destination, exist_ok=True)
        output_result_fn = utils.join(self.destination, "result.json")
        outputs = [output_result_fn]
        if self.save_snr_map:
            output_snr_map_fn = utils.join(self.destination, "snr_map.fits")
            outputs.append(output_snr_map_fn)
        self.quit_if_outputs_exist(outputs)

        aperture_diameter_px = self.aperture_diameter_px
        apertures_to_exclude = self.apertures_to_exclude

        out_image = self.image.load()
        
        start = time.perf_counter()
        result = {}

        specs = self._load_companions()
        recovered_signals = characterization.recover_signals(
            out_image, specs, aperture_diameter_px, apertures_to_exclude
        )
        result['recovered_signals'] = [dataclasses.asdict(x) for x in recovered_signals]

        if self.search is not None:
            all_candidates, (iwa_px, owa_px), snr_image = characterization.locate_snr_peaks(
                out_image, aperture_diameter_px, self.search.min_r_px, self.search.max_r_px, apertures_to_exclude, self.search.snr_threshold
            )
            result['candidates'] = [dataclasses.asdict(x) for x in all_candidates]
            result['effective_working_angles'] = {'iwa_px': iwa_px, 'owa_px': owa_px}

        end = time.perf_counter()
        time_elapsed_sec = end - start
        log.info(f"Done in {time_elapsed_sec} sec")
        result['time_elapsed_sec'] = time_elapsed_sec

        payload = {self.name: {
            'result': result,
            'config': xconf.asdict(self)
        }}
        log.info(f"Result of KLIP + ADI signal injection and recovery:")

        with fsspec.open(output_result_fn, "wb") as fh:
            payload_str = orjson.dumps(
                payload, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
            )
            fh.write(payload_str)
            fh.write(b"\n")
            log.info(payload_str.decode('utf8'))

        if self.save_snr_map:
            iofits.write_fits(iofits.PicklableHDUList([iofits.PicklableHDU(data=snr_image, kind='primary')]), output_snr_map_fn)
