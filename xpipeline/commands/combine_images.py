import sys
from xpipeline.core import LazyPipelineCollection
import fsspec.spec
import logging
import xconf


from .base import MultiInputCommand

log = logging.getLogger(__name__)

@xconf.config
class CombineImages(MultiInputCommand):
    """Combine a sequence of images into a single image"""
    operation : str = xconf.field(default='mean', help="Operation with which to combine images (so far only mean)")

    def main(self):
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
        
        # infer planes per cube
        all_inputs = self.get_all_inputs()
        hdul = iofits.load_fits_from_path(all_inputs[0])
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
        op = pipelines.CombineOperation[self.operation.upper()]
        outhdus = []
        for extkey, extshape in exts.items():
            output_hdu = pipelines.combine_extension_to_new_hdu(coll, op, extkey, extshape)
            outhdus.append(output_hdu)
        outhdus = dask.compute(outhdus)[0]
        outhdus.insert(0, iofits.DaskHDU(data=None, kind="primary"))
        outhdus[0].header.add_history(f"Combined images from {len(all_inputs)} inputs")
        return iofits.write_fits(iofits.DaskHDUList(outhdus), outname, overwrite=True)
