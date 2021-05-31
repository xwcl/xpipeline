import sys
import typing
from xpipeline.core import LazyPipelineCollection
import logging
import xconf

from .base import MultiInputCommand

log = logging.getLogger(__name__)

from .. import types

@xconf.config
class UpdateHeaders(MultiInputCommand):
    """Update FITS headers for file (See https://archive.stsci.edu/fits/fits_standard/node40.html for examples)"""
    keywords : dict[str,types.FITS_KW_VAL] = xconf.field(default_factory=dict, help="Keywords to set in extension 0")
    extensions : dict[typing.Union[str,int],dict[str,types.FITS_KW_VAL]] = xconf.field(default_factory=dict, help="Mapping of extension names to a table of KEYWORD=value pairs to set")

    def main(self):
        import fsspec.spec
        from .. import utils
        from .. import pipelines
        from ..ref import clio
        from ..tasks import iofits, sky_model, improc

        destination = self.destination
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {destination}")
        dest_fs.makedirs(destination, exist_ok=True)

        all_inputs = self.get_all_inputs()
        output_filepaths = [utils.join(destination, utils.basename(filepath)) for filepath in all_inputs]
        self.quit_if_outputs_exist(output_filepaths)

        coll = LazyPipelineCollection(all_inputs).map(iofits.load_fits_from_path)
        any_updates = False
        header_updates = {}
        if len(self.keywords):
            header_updates = {0: self.keywords}
            any_updates = True
        if 'PRIMARY' in self.extensions and 0 in header_updates:  # unlikely edge case where ext 0 is referenced by name too
            primary_updates = self.extensions.pop('PRIMARY')
            any_updates = any_updates or len(primary_updates) > 0
            header_updates[0].update(primary_updates)
        for key in self.extensions:
            any_updates = any_updates or len(self.extensions[key]) > 0
            header_updates[key] = self.extensions[key]
        if not any_updates:
            log.error("Supplied configuration would not make any modifications, exiting")
            sys.exit(0)
        output_coll = coll.map(lambda x, hdrs: x.updated_copy(new_headers_for_exts=hdrs), header_updates)
        result = output_coll.zip_map(iofits.write_fits, output_filepaths, overwrite=True).compute()
        log.info(result)

