import sys
from xpipeline.core import LazyPipelineCollection
import fsspec.spec
import logging
import xconf


from .base import MultiInputCommand, FitsConfig

log = logging.getLogger(__name__)

@xconf.config
class MagaoxCalibrate(MultiInputCommand):
    """Apply pre-processing corrections for MagAO-X PICams"""
    bias : FitsConfig = xconf.field(help="Path to full detector bias frame")

    def main(self):
        from .. import utils
        from .. import pipelines
        from ..ref import magaox
        from ..tasks import iofits, improc
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {destination}")
        dest_fs.makedirs(destination, exist_ok=True)

        # infer planes per cube
        all_inputs = self.get_all_inputs(self.input)
        hdul = iofits.load_fits_from_path(all_inputs[0])
        plane_shape = hdul[0].data.shape

        bias_arr = self.bias.load()
        if bias_arr.shape != plane_shape:
            # cut out bias subframe if needed
            xc = yc = None
            yc = None
            for key in hdul[0].header:
                if 'ROI XCEN' in key:
                    xc = hdul[0].header[key]
                if 'ROI YCEN' in key:
                    yc = hdul[0].header[key]
            if xc is None or yc is None:
                raise NotImplementedError("ROI metadata missing and no smart heuristic implemented to cut out a bias subframe")
            bbox = improc.BBox.from_center(improc.Pixel(x=xc, y=yc), extent=improc.PixelExtent(height=plane_shape[0], width=plane_shape[1]))
            bias_arr = bias_arr[bbox.slices]

        n_output_files = len(all_inputs)
        output_filepaths = [utils.join(destination, f"{self.name}_{i:04}.fits") for i in range(n_output_files)]
        self.quit_if_outputs_exist(output_filepaths)

        coll = LazyPipelineCollection(all_inputs).map(iofits.load_fits_from_path)
        output_coll = pipelines.magaox_preprocess(coll, bias_arr)
        result = output_coll.zip_map(iofits.write_fits, output_filepaths, overwrite=True).compute()
        log.info(result)
