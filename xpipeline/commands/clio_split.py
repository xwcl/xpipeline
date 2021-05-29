import sys
from xpipeline.core import LazyPipelineCollection
import fsspec.spec
import logging
import xconf

from .base import MultiInputCommand

log = logging.getLogger(__name__)

@xconf.config
class ClioSplit(MultiInputCommand):
    """Split Clio datacubes into frames and interpolate header telemetry values"""

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
        from .. import utils
        from .. import pipelines
        from ..tasks import iofits
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {destination}")
        dest_fs.makedirs(destination, exist_ok=True)

        # infer planes per cube
        all_inputs = self.get_all_inputs()
        hdul = iofits.load_fits_from_path(all_inputs[0])
        planes = hdul[0].data.shape[0]
        # plane_shape = hdul[0].data.shape[1:]

        n_output_files = len(all_inputs) * planes
        input_names = [utils.basename(fn) for fn in all_inputs]
        output_filepaths = [utils.join(destination, f"{self.name}_{i:04}.fits") for i in range(n_output_files)]
        self.quit_if_outputs_exist(output_filepaths)

        coll = LazyPipelineCollection(all_inputs).map(iofits.load_fits_from_path)
        output_coll = pipelines.clio_split(coll, input_names, frames_per_cube=planes)
        result = output_coll.zip_map(iofits.write_fits, output_filepaths, overwrite=True).compute()
        log.info(result)
