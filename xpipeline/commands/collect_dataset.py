import tqdm
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
import sys
import typing
import ray
import psutil
from dataclasses import dataclass
import logging
from astropy.io import fits
from typing import Optional, Union
import xconf
from functools import partial

from .base import MultiInputCommand
from . import base
from .. import types, utils

log = logging.getLogger(__name__)

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


@xconf.config
class SimpleObservation:
    ext : types.FITS_EXT = xconf.field(default=0, help="Extension with science data")


@xconf.config
class AdiObservation:
    angle_keyword : str = xconf.field(help="FITS header keyword with parallactic angle value")
    angle_constant : float = xconf.field(default=0.0, help="Constant offset added to angle_keyword values")
    angle_scale : float = xconf.field(default=1.0, help="Scale factor by which angle_keyword values are multiplied")


@xconf.config
class SimpleAdiObservation(AdiObservation, SimpleObservation):
    pass


@xconf.config
class VappAdiObservation(AdiObservation):
    vapp_left_ext : types.FITS_EXT = xconf.field(help=utils.unwrap(
        """FITS extension to combine across dataset containing
        complementary gvAPP-180 PSF where the dark hole is left of +Y"""
    ))
    vapp_right_ext : types.FITS_EXT = xconf.field(help=utils.unwrap(
        """FITS extension to combine across dataset containing
        complementary gvAPP-180 PSF where the dark hole is right of +Y"""
    ))

@xconf.config
class HeaderSelectionFilter:
    keyword : str = xconf.field(
        default="DATE-OBS",
        help="FITS keyword to compare"
    )
    value : typing.Union[int, str] = xconf.field(
        help="Keyword value to compare"
    )
    equal : bool = xconf.field(default=True, help="Enforce that frames have equal (true) or not equal (false) values for this keyword")
    ext : types.FITS_EXT = xconf.field(default=0, help="Extension with this header")
    # HIERARCH TWEETERSPECK SEPARATIONS == 15
    # HIERARCH HOLOOP LOOP STATE == 2

def apply_filters_to_hdulist(hdul, filters: list[HeaderSelectionFilter]):
    keep = True
    for filt in filters:
        the_val = hdul[filt.ext].header.get(filt.keyword)
        if filt.equal:
            keep = the_val == filt.value
        else:
            keep = the_val != filt.value
        if not keep:
            break
    return keep

@dataclass
class FileMetadata:
    filename : str
    header : fits.Header
    ext_shapes : dict[str, tuple[int, int]]
    ext_dtypes : dict[str, DTypeLike]

@ray.remote
def header_load_and_filter(input_fn : str, metadata_ext : Union[str, int], filters: list[HeaderSelectionFilter]) -> Optional[fits.Header]:
    with open(input_fn, 'rb') as fh:
        hdul = fits.open(fh)
        keep = apply_filters_to_hdulist(hdul, filters)
        if not keep:
            return None
        ext_shapes = {}
        ext_dtypes = {}
        for idx, exthdu in enumerate(hdul):
            if exthdu.data is not None:
                ext_shapes[exthdu.header.get('EXTNAME', idx)] = exthdu.data.shape
                ext_dtypes[exthdu.header.get('EXTNAME', idx)] = exthdu.data.dtype
        hdr = hdul[metadata_ext].header.copy()
        return FileMetadata(filename=input_fn, header=hdr, ext_shapes=ext_shapes, ext_dtypes=ext_dtypes)


@xconf.config
class CollectDataset(MultiInputCommand):
    """Convert collection of single observation files to a single multi-extension FITS file"""
    metadata_ext : typing.Union[str, int] = xconf.field(
        default=0, help="Extension from which to consolidate individual file header metadata into a table")
    date_obs_keyword : str = xconf.field(
        default="DATE-OBS", help="FITS keyword with date of observation as ISO-8601 string")
    dtype : str = xconf.field(default=None, help="Set output data type and width (default: use input)")
    obs : typing.Union[SimpleObservation, SimpleAdiObservation, VappAdiObservation] = xconf.field(
        default_factory=SimpleObservation,
        help="Observation strategy metatdata"
    )
    filters : list[HeaderSelectionFilter] = xconf.field(
        default_factory=list,
        help="Filters to remove frames from the collection"
    )
    ray : base.AnyRayConfig = xconf.field(
        default_factory=base.LocalRayConfig,
        help="Ray distributed framework configuration"
    )
    batch_size : int = xconf.field(default=32)

    def main(self):
        # import dask
        # from ray.util.dask import ray_dask_get
        self.ray.init()
        # from xpipeline.core import LazyPipelineCollection
        import fsspec.spec
        import numpy as np
        from .. import utils
        from .. import pipelines
        from ..ref import clio
        from ..tasks import iofits, vapp, obs_table

        destination = self.destination
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {destination}")
        dest_fs.makedirs(destination, exist_ok=True)

        all_inputs = self.get_all_inputs(self.input)
        output_filepath = utils.join(destination, utils.basename("collect_dataset.fits"))
        self.quit_if_outputs_exist([output_filepath])

        log.debug(f"Memory avail: {psutil.virtual_memory().available / 1024 / 1024} MB")

        # parallel load/parse headers
        futs = []
        for fn in all_inputs:
            futs.append(header_load_and_filter.remote(fn, self.metadata_ext, self.filters))
        metas: list[FileMetadata] = []
        skipped = 0
        with tqdm.tqdm(desc='load headers', total=len(all_inputs)) as pbar:
            while futs:
                finished_futs, futs = ray.wait(futs, num_returns=min(self.batch_size, len(futs)))
                new_metas = ray.get(finished_futs)
                for m in new_metas:
                    if m is not None:
                        metas.append(m)
                    else:
                        skipped += 1
                pbar.update(len(finished_futs))


        log.info(f"Collecting {len(metas)} frames")
        if len(self.filters):
            log.info(f"Skipped {skipped}")

        log.debug(f"Memory avail: {psutil.virtual_memory().available / 1024 / 1024} MB")

        # sort metadata by date
        metas.sort(key=lambda x: x.header[self.date_obs_keyword])

        # figure out the memory requirements
        example_meta = metas[0]
        num_files_kept = len(metas)
        total_size_bytes = 0
        extnames = tuple(example_meta.ext_dtypes.keys())
        for k in extnames:
            shape = example_meta.ext_shapes[k]
            dtype = example_meta.ext_dtypes[k]
            # seems like there should be a better way to introspect itemsize...
            size_bytes = np.prod(shape) * np.ones(1, dtype=dtype).itemsize
            log.debug(f"extname: {k} -- {shape} <{dtype}> ({size_bytes/1024/1024} MB)")
            total_size_bytes += size_bytes
        log.info(f"Array data total: {total_size_bytes / 1024 / 1024} MB")
        stats = psutil.virtual_memory()  # returns a named tuple
        available = getattr(stats, 'available')
        if total_size_bytes > 0.8 * available:
            log.warning(f"Allocation may fail, {available / 1024 / 1024} MB available")
        if total_size_bytes > available:
            raise RuntimeError(f"Need at least {total_size_bytes} bytes RAM, but OS reports {available} bytes free")

        # make a metadata table from the headers
        obs_table_name = "OBSTABLE"
        obs_table_mask_name = "OBSTABLE_MASK"

        log.debug(f"Memory avail: {psutil.virtual_memory().available / 1024 / 1024} MB")
        log.info("Constructing metadata table...")
        static_header, metadata_table = obs_table.construct_headers_table([x.header for x in metas])
        log.debug(f"Memory avail: {psutil.virtual_memory().available / 1024 / 1024} MB")

        date_obs_keyword = self.date_obs_keyword
        obs_method = {}

        # compute derotation angles?
        if hasattr(self.obs, 'angle_keyword'):
            angle_keyword = self.obs.angle_keyword
            angle_scale = self.obs.angle_scale
            angle_constant = self.obs.angle_constant
            derotation_angles = (
                angle_scale * metadata_table[angle_keyword] + angle_constant
            )
            metadata_table = np.lib.recfunctions.append_fields(
                metadata_table, "derotation_angle_deg", derotation_angles
            )
            if "adi" not in obs_method:
                obs_method["adi"] = {}
            obs_method["adi"][
                "derotation_angles"
            ] = f"{obs_table_name}.derotation_angle_deg"

        static_header["OBSMETHD"] = utils.flatten_obs_method(obs_method)
        log.debug(f"OBSMETHD {static_header['OBSMETHD']}")

        hdul = iofits.PicklableHDUList(
            [iofits.PicklableHDU(data=None, header=static_header, kind="primary")]
        )
        cubes = {}
        for extname, shape in example_meta.ext_shapes.items():
            cubes[extname] = np.zeros((num_files_kept,) + shape, dtype=example_meta.ext_dtypes[extname])
        
        for idx, meta in enumerate(tqdm.tqdm(metas, desc='copy data')):
            with open(meta.filename, 'rb') as fh:
                hdul = fits.open(fh)
                for extname in cubes:
                    cubes[extname][idx] = hdul[extname].data

        for extname, cube in cubes.items():
            outcube = cube
            if self.dtype is not None:
                output_dtype = _DTYPE_LOOKUP[self.dtype]
                outcube = outcube.astype(output_dtype)
            hdu = iofits.PicklableHDU(outcube)
            hdu.header["EXTNAME"] = extname
            hdul.append(hdu)

        table_hdu = iofits.PicklableHDU(metadata_table, kind="bintable")
        table_hdu.header["EXTNAME"] = obs_table_name
        hdul.append(table_hdu)

        table_mask_hdu = iofits.PicklableHDU(metadata_table.mask, kind="bintable")
        table_mask_hdu.header["EXTNAME"] = obs_table_mask_name
        hdul.append(table_mask_hdu)

        iofits.write_fits(hdul, output_filepath, overwrite=True)
