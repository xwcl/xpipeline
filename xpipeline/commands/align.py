import sys
from xpipeline.core import LazyPipelineCollection
import fsspec.spec
import logging
import typing
import xconf


from .. import utils
from . import base

log = logging.getLogger(__name__)

@xconf.config
class Box:
    height : int = xconf.field(default=None, help="Height of search box")
    width : int = xconf.field(default=None, help="Width of search box")

    def to_extent(self):
        from ..tasks import improc
        height, width = self.height, self.width
        return improc.PixelExtent(width=width, height=height)

@xconf.config
class BoxFromCenter(Box):
    center_x : int
    center_y : int

    def to_bbox(self):
        from ..tasks import improc
        extent = self.to_extent()
        return improc.BBox.from_center(center=improc.Pixel(y=self.center_y, x=self.center_x), extent=extent)

@xconf.config
class BoxFromOrigin(Box):
    origin_y : int
    origin_x : int

    def to_bbox(self):
        from ..tasks import improc
        extent = self.to_extent()
        yo, xo = self.origin_y, self.origin_x
        return improc.BBox(
            origin=improc.Pixel(x=xo, y=yo),
            extent=extent
        )

@xconf.config
class Align(base.MultiInputCommand):
    "Align images to common center"
    registration_regions : list[typing.Union[BoxFromCenter,BoxFromOrigin,Box]] = xconf.field(default=Box(), help="Search box to find the PSF to cross-correlate on")
    ext : typing.Union[str, int] = xconf.field(default=0, help="Extension index or name to load from input files")
    excluded_regions : typing.Optional[base.FileConfig] = xconf.field(default_factory=None, help="Regions to fill with zeros before cross-registration, stored as DS9 region file (reg format)")

    def main(self):
        import dask
        from .. import pipelines
        from ..tasks import iofits, improc, regions

        log.debug(self)
        dest_fs = utils.get_fs(self.destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {self.destination}")
        dest_fs.makedirs(self.destination, exist_ok=True)

        all_inputs = self.get_all_inputs()
        n_output_files = len(all_inputs)
        output_filepaths = [utils.join(self.destination, f"aligned_{i:04}.fits") for i in range(n_output_files)]
        self.quit_if_outputs_exist(output_filepaths)

        input_coll = LazyPipelineCollection(all_inputs)
        coll = input_coll.map(iofits.load_fits_from_path)
        example_hdul = dask.compute(coll.items[0])[0]
        example_data = example_hdul[self.ext].data
        dimensions = example_data.shape

        if isinstance(self.excluded_regions, base.FileConfig):
            with self.excluded_regions.open() as fh:
                excluded_regions = regions.load_file(fh)
            excluded_pixels_mask = regions.make_mask(excluded_regions, dimensions)
        else:
            excluded_pixels_mask = np.zeros(dimensions, dtype=bool)

        regions = {}
        for region in self.registration_regions:
            bbox = region.to_bbox()
            region[bbox] = example_data[bbox.slices]
        
        output_coll = pipelines.align_to_templates(coll, cutout_specs, excluded_pixels_mask)
        res = output_coll.zip_map(iofits.write_fits, output_filepaths, overwrite=True)

        result = res.compute()
        log.info(result)

    def _search_box_to_bbox(self, search_box, default_height, default_width):
        
