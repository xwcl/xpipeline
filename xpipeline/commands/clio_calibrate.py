import sys
from xpipeline.core import LazyPipelineCollection
import fsspec.spec
import logging
import xconf


from .base import MultiInputCommand

log = logging.getLogger(__name__)

@xconf.config
class ClioCalibrate(MultiInputCommand):
    """Apply bad pixel map, linearity correction, and saturation flags"""
    badpix_path : str = xconf.field(help="Path to full detector bad pixel map FITS file (1 where pixel is bad)")

    def main(self):
        from .. import utils
        from .. import pipelines
        from ..ref import clio
        from ..tasks import iofits
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {destination}")
        dest_fs.makedirs(destination, exist_ok=True)

        # infer planes per cube
        all_inputs = self.get_all_inputs()
        hdul = iofits.load_fits_from_path(all_inputs[0])
        plane_shape = hdul[0].data.shape

        n_output_files = len(all_inputs)
        output_filepaths = [utils.join(destination, f"{self.name}_{i:04}.fits") for i in range(n_output_files)]
        for output_file in output_filepaths:
            if dest_fs.exists(output_file):
                log.error(f"Output exists: {output_file}")
                sys.exit(1)

        coll = LazyPipelineCollection(all_inputs).map(iofits.load_fits_from_path)
        badpix_path = self.badpix_path
        full_badpix_arr = iofits.load_fits_from_path(badpix_path)[0].data
        badpix_arr = clio.badpix_for_shape(full_badpix_arr, plane_shape)
        output_coll = pipelines.clio_badpix_linearity(coll, badpix_arr, plane_shape)
        return output_coll.zip_map(iofits.write_fits, output_filepaths, overwrite=True).compute()

