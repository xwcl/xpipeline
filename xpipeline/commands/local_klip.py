from astropy.io import fits
import numpy as np
import dask.array as da
import os.path
import logging
from ..utils import unwrap
from .. import utils
from ..tasks import iofits, improc, starlight_subtraction # obs_table, iofits, sky_model, detector, data_quality

from .klip import KLIP

log = logging.getLogger(__name__)


class LocalKLIP(KLIP):
    name = "local_klip"
    help = "Single-machine starlight subtraction"

    def main(self):
        destination = self.args.destination
        os.makedirs(destination, exist_ok=True)

        output_klip_final = utils.join(destination, "klip_final.fits")
        if self.check_for_outputs([output_klip_final]):
            return
        
        inputs = []
        for filepath in self.all_files:
            with open(filepath, 'rb') as fh:
                hdul = fits.open(fh)
                inputs.append(iofits.DaskHDUList.from_fits(hdul))
        

        if self.args.region_mask is not None:
            with open(self.args.region_mask, 'rb') as fh:
                region_mask = fits.open(fh)[0].data
        else:
            region_mask = np.ones_like(inputs[0][0].data)
        plane_shape = region_mask.shape
        rotation_keyword = self.args.angle_keyword
        rotation_scale = self.args.angle_scale
        rotation_offset = self.args.angle_constant
        sci_arr = np.stack([x[0].data.astype('=f8') for x in inputs])

        rot_arr = np.asarray([x[0].header[rotation_keyword] for x in inputs])
        derotation_angles = rotation_scale * rot_arr + rotation_offset
        
        log.info(f"Computing klip pipeline result...")
        import time
        start = time.perf_counter()
        mtx_x, x_indices, y_indices = improc.unwrap_cube(sci_arr, region_mask)

        subtracted_mtx = starlight_subtraction.klip_cube(
            mtx_x,
            self.args.k_klip,
            self.args.exclude_nearest_n_frames
        )
        outcube = improc.wrap_matrix(subtracted_mtx, sci_arr.shape, x_indices, y_indices)
        out_image = improc.quick_derotate(outcube, derotation_angles)
        elapsed = time.perf_counter() - start
        log.info(f"Computed in {elapsed} sec")
        out_hdul = iofits.DaskHDUList([iofits.DaskHDU(out_image)])
        iofits.write_fits(out_hdul, output_klip_final)
