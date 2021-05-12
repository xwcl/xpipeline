import sys
from astropy.io import fits
import numpy as np
import argparse
import dask
import fsspec.spec
import os.path
import logging

# from . import constants as const
from ..utils import unwrap
from .. import utils
from .. import pipelines  # , irods
from ..core import LazyPipelineCollection
from ..tasks import iofits, vapp  # obs_table, iofits, sky_model, detector, data_quality

# from .ref import clio

from .base import MultiInputCommand

log = logging.getLogger(__name__)


def _optflag_to_attr(optflag):
    return optflag.replace("--", "").replace("-", "_")


_DTYPE_LOOKUP = {
    "float32": np.float32,
    "float64": np.float64,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
}


class CollectDataset(MultiInputCommand):
    name = "collect_dataset"
    help = "Collect matching FITS files into a single data file"

    _keyword_override_options = {
        "--telescope": {
            "keyword": "TELESCOP",
            "help": "Name of telescope where data were taken",
        },
        "--instrument": {
            "keyword": "INSTRUME",
            "help": "Name of instrument with which data were taken",
        },
        "--observer": {
            "keyword": "OBSERVER",
            "help": "Name of observer",
        },
        "--object": {
            "keyword": "OBJECT",
            "help": "Name object observed",
        },
    }

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--ext",
            action="append",
            default=None,
            help=unwrap(
                """
                FITS extension(s) to combine across dataset (can be
                repeated to make multiple separate cubes from different
                extension names)
            """
            ),
        )
        parser.add_argument(
            "--vapp-left-ext",
            default=None,
            help=unwrap(
                """
                FITS extension to combine across dataset containing
                complementary gvAPP-180 PSF where the dark hole is left of +Y
            """
            ),
        )
        parser.add_argument(
            "--vapp-right-ext",
            default=None,
            help=unwrap(
                """
                FITS extension to combine across dataset containing
                complementary gvAPP-180 PSF where the dark hole is right of +Y
            """
            ),
        )
        parser.add_argument(
            "--metadata-ext",
            default=None,
            help="FITS extension with observation metadata to extract into a table",
        )
        parser.add_argument(
            "--angle-keyword",
            default=None,
            help="FITS keyword with angle for ADI observations",
        )
        parser.add_argument(
            "--angle-scale",
            type=float,
            default=1.0,
            help="Scale factor relating keyword value to angle in degrees needed to rotate image North-up (default: 1.0)",
        )
        parser.add_argument(
            "--angle-constant",
            type=float,
            default=0.0,
            help="Constant factor added to (scale * keyword value) to get angle in degrees needed to rotate image North-up (default: 0.0)",
        )
        parser.add_argument(
            "--date-obs-keyword",
            default="DATE-OBS",
            help="FITS keyword with date of observation as ISO-8601 string (default: 'DATE-OBS')",
        )
        parser.add_argument(
            "-M",
            "--obs-method",
            default=None,
            action="append",
            help="key=value hints about data interpretation to store in a 'OBSMETHD' header in the combined file, can be repeated",
        )
        parser.add_argument(
            "--dtype",
            default=None,
            choices=list(_DTYPE_LOOKUP.keys()),
            help="Set output data type and width (default: use input)",
        )
        for optflag, info in CollectDataset._keyword_override_options.items():
            parser.add_argument(optflag, default=None, help=info["help"])
        return super(CollectDataset, CollectDataset).add_arguments(parser)

    def _normalize_extension_key(self, key):
        try:
            return int(key)
        except ValueError:
            return key

    def _normalize_extension_keys(self, keys):
        out = []
        for k in keys:
            out.append(self._normalize_extension_key(k))
        return out

    def main(self):
        destination = self.args.destination
        log.debug(f"{destination=}")
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {destination}")
        dest_fs.makedirs(destination, exist_ok=True)

        output_file = utils.join(destination, "collected_dataset.fits")
        if dest_fs.exists(output_file):
            log.error(f"Output exists: {output_file}")
            sys.exit(1)

        obs_method = (
            utils.parse_obs_method(" ".join(self.args.obs_method))
            if self.args.obs_method is not None
            else {}
        )
        extension_keys = (
            self._normalize_extension_keys(self.args.ext)
            if self.args.ext is not None
            else []
        )
        if self.args.vapp_left_ext is not None or self.args.vapp_right_ext is not None:
            if self.args.vapp_left_ext is None:
                log.error(
                    f"Got --vapp-right-ext {self.args.vapp_right_ext} but no --vapp-left-ext"
                )
                sys.exit(1)
            if self.args.vapp_right_ext is None:
                log.error(
                    f"Got --vapp-left-ext {self.args.vapp_left_ext} but no --vapp-right-ext"
                )
                sys.exit(1)
            left_extname = self._normalize_extension_key(self.args.vapp_left_ext)
            if left_extname not in extension_keys:
                extension_keys.append(left_extname)
            right_extname = self._normalize_extension_key(self.args.vapp_right_ext)
            if right_extname not in extension_keys:
                extension_keys.append(right_extname)
            if not "vapp" in obs_method:
                obs_method["vapp"] = {}
            obs_method["vapp"]["left"] = left_extname
            obs_method["vapp"]["right"] = right_extname
        else:
            left_extname = right_extname = None

        if len(extension_keys) == 0:
            extension_keys = [0]
        log.debug(f"{extension_keys=}")

        inputs = self.inputs_coll.map(iofits.load_fits_from_path)
        log.debug(f"{inputs.items=}")
        (first,) = dask.compute(inputs.items[0])
        n_inputs = len(inputs.items)
        cubes = {}
        for ext in extension_keys:
            if isinstance(ext, int):
                rewrite_ext = "SCI" if ext == 0 else "SCI_{ext}"
            else:
                rewrite_ext = ext
            if rewrite_ext in cubes:
                raise RuntimeError(
                    f"Name collision {ext=} {rewrite_ext=} {list(cubes.keys())=}"
                )
            cubes[rewrite_ext] = iofits.hdulists_to_dask_cube(
                inputs.items,
                ext=ext,
                plane_shape=first[ext].data.shape,
                dtype=first[ext].data.dtype,
            )

        obs_table_name = "OBSTABLE"

        metadata_ext = (
            self._normalize_extension_key(self.args.metadata_ext)
            if self.args.metadata_ext is not None
            else 0
        )
        static_header, metadata_table = inputs.map(
            lambda hdulist: hdulist[metadata_ext].header.copy()
        ).collect(iofits.construct_headers_table, _delayed_kwargs={"nout": 2})
        static_header, metadata_table, cubes = dask.compute(
            static_header, metadata_table, cubes
        )

        date_obs_keyword = self.args.date_obs_keyword
        sorted_index = np.argsort(metadata_table[date_obs_keyword])
        metadata_table = metadata_table[sorted_index]

        # compute derotation angles?
        angle_keyword = self.args.angle_keyword
        angle_scale = self.args.angle_scale
        angle_constant = self.args.angle_constant

        if angle_keyword is not None:
            derotation_angles = (
                angle_scale * metadata_table[angle_keyword] + angle_constant
            )
            metadata_table = np.lib.recfunctions.append_fields(
                metadata_table, "derotation_angle_deg", derotation_angles
            )
            if "ADI" not in obs_method:
                obs_method["adi"] = {}
            obs_method["adi"][
                "derotation_angles"
            ] = f"{obs_table_name}.derotation_angle_deg"

        static_header["OBSMETHD"] = utils.flatten_obs_method(obs_method)

        for optflag, info in CollectDataset._keyword_override_options.items():
            optval = getattr(self.args, _optflag_to_attr(optflag))
            if optval is None:
                if info["keyword"] not in static_header:
                    log.warn(
                        f"No '{info['keyword']}' header found and no {optflag} option provided"
                    )
            else:
                if info["keyword"] in static_header:
                    log.warn(
                        f"Both '{info['keyword']}' header ({static_header[info['keyword']]}) and {optflag} option ({optval}) provided, using option value"
                    )
                static_header[info["keyword"]] = optval

        hdul = iofits.DaskHDUList(
            [iofits.DaskHDU(data=None, header=static_header, kind="primary")]
        )

        # handle paired vAPP cubes if requested
        if left_extname is not None:
            log.debug(
                f"Before crop: {cubes[left_extname].shape=} {cubes[right_extname].shape=}"
            )
            cubes[left_extname], cubes[right_extname] = vapp.crop_paired_cubes(
                cubes[left_extname], cubes[right_extname]
            )
            log.debug(
                f"After crop: {cubes[left_extname].shape=} {cubes[right_extname].shape=}"
            )

        for extname, cube in cubes.items():
            outcube = cube[sorted_index]
            if self.args.dtype is not None:
                output_dtype = _DTYPE_LOOKUP[self.args.dtype]
                outcube = outcube.astype(output_dtype)
            hdu = iofits.DaskHDU(outcube)
            hdu.header["EXTNAME"] = extname
            hdul.append(hdu)
        table_hdu = iofits.DaskHDU(metadata_table, kind="bintable")
        table_hdu.header["EXTNAME"] = obs_table_name
        hdul.append(table_hdu)

        return iofits.write_fits(hdul, output_file, overwrite=True)
