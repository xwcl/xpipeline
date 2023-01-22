import numpy as np
import sys
import typing
import logging
import xconf

from .base import MultiInputCommand
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
class CollectDataset(MultiInputCommand):
    """Convert collection of single observation files to a single multi-extension FITS file"""
    metadata_ext : typing.Union[str, int] = xconf.field(
        default=0, help="Extension from which to consolidate individual file header metadata into a table")
    date_obs_keyword : str = xconf.field(
        default="DATE-OBS", help="FITS keyword with date of observation as ISO-8601 string")
    dtype : str = xconf.field(default=None, help="Set output data type and width (default: use input)")
    obs : typing.Union[SimpleObservation, SimpleAdiObservation, VappAdiObservation] = xconf.field(
        default=SimpleObservation(),
        help="Observation strategy metatdata"
    )

    def main(self):
        import dask
        from xpipeline.core import LazyPipelineCollection
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

        all_inputs = self.get_all_inputs()
        output_filepath = utils.join(destination, utils.basename("collect_dataset.fits"))
        self.quit_if_outputs_exist([output_filepath])

        inputs = LazyPipelineCollection(all_inputs).map(iofits.load_fits_from_path)
        (first,) = dask.compute(inputs.items[0])
        n_inputs = len(inputs.items)
        cubes = {}
        for ext in first.extnames:
            if isinstance(ext, int):
                rewrite_ext = "SCI" if ext == 0 else "SCI_{ext}"
            else:
                rewrite_ext = ext
            if rewrite_ext in cubes:
                raise RuntimeError(
                    f"Name collision {ext=} {rewrite_ext=} {list(cubes.keys())=}"
                )
            if not len(first[ext].data.shape):
                continue
            cubes[rewrite_ext] = iofits.hdulists_to_dask_cube(
                inputs.items,
                ext=ext,
                plane_shape=first[ext].data.shape,
                dtype=first[ext].data.dtype,
            )

        obs_table_name = "OBSTABLE"
        obs_table_mask_name = "OBSTABLE_MASK"

        static_header, metadata_table = (inputs
            .map(lambda hdulist: hdulist[self.metadata_ext].header.copy())
            .collect(obs_table.construct_headers_table, _delayed_kwargs={"nout": 2})
        )

        static_header, metadata_table, cubes = dask.compute(
            static_header, metadata_table, cubes
        )

        date_obs_keyword = self.date_obs_keyword
        sorted_index = np.argsort(metadata_table[date_obs_keyword])
        metadata_table = metadata_table[sorted_index]

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

        

        # handle paired vAPP cubes if requested
        if hasattr(self.obs, 'vapp_left_ext'):
            left_extname = self.obs.vapp_left_ext
            right_extname = self.obs.vapp_right_ext
            log.debug(
                f"Before crop: {cubes[left_extname].shape=} {cubes[right_extname].shape=}"
            )
            cubes[left_extname], cubes[right_extname] = vapp.crop_paired_cubes(
                cubes[left_extname], cubes[right_extname]
            )
            log.debug(
                f"After crop: {cubes[left_extname].shape=} {cubes[right_extname].shape=}"
            )
            if "vapp" not in obs_method:
                obs_method['vapp'] = {}
            obs_method['vapp']['left'] = left_extname
            obs_method['vapp']['right'] = right_extname
        elif hasattr(self.obs, "ext"):
            obs_method["ext"] = self.obs.ext if self.obs.ext != 0 else "SCI"
        else:
            raise RuntimeError(f"No extension specified as ext= or vapp_*_ext= (shouldn't happen at this point)")

        static_header["OBSMETHD"] = utils.flatten_obs_method(obs_method)
        log.debug(f"OBSMETHD {static_header['OBSMETHD']}")

        hdul = iofits.DaskHDUList(
            [iofits.DaskHDU(data=None, header=static_header, kind="primary")]
        )
        for extname, cube in cubes.items():
            outcube = cube[sorted_index]
            if self.dtype is not None:
                output_dtype = _DTYPE_LOOKUP[self.dtype]
                outcube = outcube.astype(output_dtype)
            hdu = iofits.DaskHDU(outcube)
            hdu.header["EXTNAME"] = extname
            hdul.append(hdu)

        table_hdu = iofits.DaskHDU(metadata_table, kind="bintable")
        table_hdu.header["EXTNAME"] = obs_table_name
        hdul.append(table_hdu)

        table_mask_hdu = iofits.DaskHDU(metadata_table.mask, kind="bintable")
        table_mask_hdu.header["EXTNAME"] = obs_table_mask_name
        hdul.append(table_mask_hdu)

        iofits.write_fits(hdul, output_filepath, overwrite=True)
