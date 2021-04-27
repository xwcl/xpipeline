import sys
from pprint import pformat
from astropy.io import fits
import numpy as np
import argparse
import dask
import dask.array as da
import fsspec.spec
import os.path
import logging
import orjson
# from . import constants as const
from ..utils import unwrap
from .. import utils
from .. import pipelines #, irods
from ..core import LazyPipelineCollection
from ..tasks import iofits, characterization
# from .ref import clio
from .klip import KLIP

from .base import BaseCommand

log = logging.getLogger(__name__)

def _docs_args(parser):
    # needed for sphinx-argparse support
    return EvalKLIP.add_arguments(parser)

class EvalKLIP(KLIP):
    name = "eval_klip"
    help = "Inject and recover a companion in ADI data through KLIP"

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "template_psf",
            help=unwrap("""
                Path to FITS image of template PSF, scaled to the
                average amplitude of the host star signal such that
                multiplying by the contrast gives an appropriately
                scaled planet PSF
            """)
        )
        parser.add_argument(
            "--companion-spec",
            action='append',
            required=True,
            help=unwrap("""
                specification of the form ``scale,r,theta``, can be
                repeated (e.g. ``0.0001,34,100`` for 10^-4 contrast,
                34 px, 100 deg E of N). Negative injections are
                supported too (to remove biasing effect of true
                companions) as is a scale of ``?`` as a shorthand for
                zero-scaled specs (for measuring true SNR without
                injection)
            """)
        )
        parser.add_argument(
            "--aperture-diameter-px",
            type=float,
            required=True,
            help=unwrap("""
                Diameter of the SNR estimation aperture (~lambda/D) in pixels
            """)
        )
        parser.add_argument(
            "--apertures-to-exclude",
            help=unwrap("""
                Number of apertures on *each side* of the specified target
                location to exclude when calculating the noise (in other
                words, a value of 1 excludes two apertures total,
                one on each side)
            """),
            default=2
        )
        return super(EvalKLIP, EvalKLIP).add_arguments(parser)

    def _load_inputs(self):
        sci_arr, rot_arr, region_mask = super()._load_inputs()
        template_psf = iofits.load_fits_from_path(self.args.template_psf)[0].data
        return sci_arr, rot_arr, region_mask, template_psf

    def _load_companions(self):
        specs = []
        for spec_str in self.args.companion_spec:
            try:
                specs.append(characterization.CompanionSpec.from_str(spec_str))
            except ValueError:
                log.error(f"Couldn't parse {spec_str} into scale, r_px, and pa_deg")
                sys.exit(1)
        return specs

    def main(self):
        output_result = utils.join(self.destination, "result.json")
        if self.check_for_outputs([output_result]):
            return
        specs = self._load_companions()

        aperture_diameter_px = self.args.aperture_diameter_px
        apertures_to_exclude = self.args.apertures_to_exclude
        k_klip = self.args.k_klip
        exclude_nearest_n_frames = self.args.exclude_nearest_n_frames

        sci_arr, derotation_angles, region_mask, template_psf = self._load_inputs()

        d_recovered_signals = pipelines.evaluate_starlight_subtraction(
            sci_arr,
            derotation_angles,
            region_mask,
            specs,
            template_psf,
            self.args.angle_scale,
            self.args.angle_constant,
            exclude_nearest_n_frames,
            k_klip,
            aperture_diameter_px,
            apertures_to_exclude,
        )
        recovered_signals = dask.compute(d_recovered_signals)[0]
        payload = {
            'inputs': self.all_files,
            'template_psf': self.args.template_psf,
            'k_klip': k_klip,
            'exclude_nearest_n_frames': exclude_nearest_n_frames,
            'aperture_diameter_px': aperture_diameter_px,
            'apertures_to_exclude': apertures_to_exclude,
            'region_mask': self.args.region_mask,
            'companions': [
                {'scale': rs.scale, 'r_px': rs.r_px, 'pa_deg': rs.pa_deg, 'snr': rs.snr}
                for rs in recovered_signals
            ],
        }
        log.info(f'Result of KLIP + ADI signal injection and recovery:')
        log.info(pformat(payload))
        with fsspec.open(output_result, 'wb') as fh:
            payload_str = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2)
            fh.write(payload_str)
            fh.write(b'\n')

        return output_result
