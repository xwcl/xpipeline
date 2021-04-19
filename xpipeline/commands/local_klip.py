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
        output_exptime_map = utils.join(destination, "klip_exptime_map.fits")
        if self.check_for_outputs([output_klip_final, output_exptime_map]):
            return
        
        inputs = []
        if len(self.all_files) == 1:
            log.debug(f'Loading cube from {self.all_files[0]}')
            extname = self.args.extname
            rotation_extname = self.args.angle_extname
            with open(self.all_files[0], 'rb') as fh:
                hdul = fits.open(fh)
                sci_arr = hdul[extname].data.astype('=f8')[::self.args.sample]
                rot_arr = hdul[rotation_extname].data.astype('=f8')[::self.args.sample]
        else:
            for filepath in self.all_files:
                with open(filepath, 'rb') as fh:
                    log.debug(f'Loading frame from {filepath}')
                    hdul = fits.open(fh)
                    inputs.append(iofits.DaskHDUList.from_fits(hdul))
            sci_arr = np.stack([x[0].data.astype('=f8') for x in inputs])
            rotation_keyword = self.args.angle_keyword
            rot_arr = np.asarray([x[0].header[rotation_keyword] for x in inputs])

        if self.args.region_mask is not None:
            with open(self.args.region_mask, 'rb') as fh:
                region_mask = fits.open(fh)[0].data
        else:
            region_mask = np.ones_like(sci_arr[0])
        plane_shape = region_mask.shape
        rotation_scale = self.args.angle_scale
        rotation_offset = self.args.angle_constant

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
        fits.PrimaryHDU(out_image).writeto(output_klip_final, overwrite=True)
        log.info(f'Wrote result to {output_klip_final}')
        return output_klip_final
