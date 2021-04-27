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
from .. import pipelines #, irods
from ..core import LazyPipelineCollection
from ..tasks import iofits # obs_table, iofits, sky_model, detector, data_quality
# from .ref import clio

from .base import MultiInputCommand

log = logging.getLogger(__name__)


class CollectDataset(MultiInputCommand):
    name = "collect_dataset"
    help = "Collect matching FITS files into a single data file"

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            '--ext',
            action='append',
            default=None,
            help="FITS extension(s) to combine across dataset (can be repeated)"
        )
        parser.add_argument(
            '--metadata-ext',
            default=None,
            help="FITS extension with observation metadata to extract into a table"
        )
        parser.add_argument(
            '--obs-method',
            default=None,
            help="Hints about data interpretation to store in a 'OBSMETHD' header in the combined file"
        )
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
        log.debug(f'{destination=}')
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f'calling makedirs on {dest_fs} at {destination}')
        dest_fs.makedirs(destination, exist_ok=True)
        extension_keys = self._normalize_extension_keys(self.args.ext) if self.args.ext is not None else [0]
        log.debug(f'{extension_keys=}')
        output_file = utils.join(destination, "collected_dataset.fits")
        if dest_fs.exists(output_file):
            log.error(f"Output exists: {output_file}")
            sys.exit(1)

        inputs = self.inputs_coll.map(iofits.load_fits_from_path)
        log.debug(f'{inputs.collection=}')
        dask.persist(inputs.collection)
        first, = dask.compute(inputs.collection[0])
        n_inputs = len(inputs.collection)
        cubes = {}
        for ext in extension_keys:
            if isinstance(ext, int):
                rewrite_ext = 'SCI' if ext == 0 else 'SCI_{ext}'
            else:
                rewrite_ext = ext
            if rewrite_ext in cubes:
                raise RuntimeError(f"Name collision {ext=} {rewrite_ext=} {list(cubes.keys())=}")
            cubes[rewrite_ext] = iofits.hdulists_to_dask_cube(
                inputs.collection,
                ext=ext,
                plane_shape=first[ext].data.shape,
                dtype=first[ext].data.dtype
            )
        
        metadata_ext = (
            self._normalize_extension_key(self.args.metadata_ext)
            if self.args.metadata_ext is not None
            else 0
        )
        static_header, metadata_table = inputs.map(
            lambda hdulist: hdulist[metadata_ext].header.copy()
        ).collect(
            iofits.construct_headers_table,
            _delayed_kwargs={'nout': 2}
        ).persist()

        hdul = iofits.DaskHDUList([iofits.DaskHDU(data=None, header=static_header.compute(), kind='primary')])
        for extname, cube in cubes.items():
            hdu = iofits.DaskHDU(cube.compute())
            hdu.header['EXTNAME'] = extname
            hdul.append(hdu)
        table_hdu = iofits.DaskHDU(metadata_table.compute(), kind='bintable')
        table_hdu.header['EXTNAME'] = 'OBSTABLE'
        hdul.append(table_hdu)
        return iofits.write_fits(hdul, output_file, overwrite=True)

