import sys
from xpipeline.core import LazyPipelineCollection
import fsspec.spec
import logging
import xconf


from .base import MultiInputCommand

from ..constants import CombineOperation

log = logging.getLogger(__name__)

@xconf.config
class CombineImages(MultiInputCommand):
    """Combine a sequence of images into a single image"""
    operation : CombineOperation = xconf.field(default=CombineOperation.MEAN, help="Operation with which to combine images")

    def main(self):
        import numpy as np
        import dask
        from .. import utils
        from .. import pipelines
        from ..tasks import iofits
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {destination}")
        dest_fs.makedirs(destination, exist_ok=True)

        outname = utils.join(destination, 'combine_images.fits')
        self.quit_if_outputs_exist([outname])

        all_inputs = self.get_all_inputs()
        hdul = iofits.load_fits_from_path(all_inputs[0])
        if len(all_inputs) > 1:
            # infer planes per cube
            exts = {}
            log.debug('before loop')
            for idx, extension in enumerate(hdul):
                log.debug(f'{extension.header}')
                if len(extension.data.shape) != 2:
                    continue
                if extension.header['XTENSION'] != 'IMAGE':
                    continue
                if 'EXTNAME' in extension.header:
                    key = extension.header['EXTNAME']
                else:
                    key = idx
                exts[key] = extension.data.shape
            log.debug('after loop')
            if len(exts) == 0:
                raise RuntimeError(f"Examining {all_inputs[0]} found no suitable image extensions to combine")
            log.debug(f'Got exts/shapes {exts}')

            coll = LazyPipelineCollection(all_inputs).map(iofits.load_fits_from_path)
            outhdus = []
            for extkey, extshape in exts.items():
                output_hdu = pipelines.combine_extension_to_new_hdu(coll, op, extkey, extshape)
                outhdus.append(output_hdu)
            outhdus = dask.compute(outhdus)[0]
            outhdus.insert(0, iofits.DaskHDU(data=None, kind="primary"))
            outhdus[0].header.add_history(f"Combined images from {len(all_inputs)} inputs")
            return iofits.write_fits(iofits.DaskHDUList(outhdus), outname, overwrite=True)
        else:
            new_data_for_exts = {}
            for extname in hdul.extnames:
                if hdul[extname].kind != 'image' or len(hdul[extname].data.shape) == 0:
                    continue
                # import IPython
                # IPython.embed()
                if op is pipelines.CombineOperation.MEAN:
                    new_data_for_exts[extname] = np.nanmean(hdul[extname].data, axis=0)
                else:
                    raise ValueError("Unknown operation")

            new_hdul = hdul.updated_copy(new_data_for_exts=new_data_for_exts, history=f"Combined cube using {op}")
            return iofits.write_fits(new_hdul, outname, overwrite=True)
