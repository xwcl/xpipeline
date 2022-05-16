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
class GaussianTemplate:
    sigma_px : float = xconf.field(default=1, help="Template PSF kernel stddev in pixels")
    size_px : int = xconf.field(default=128, help="Size of generated Gaussian template in pixels")

@xconf.config
class Box:
    height : typing.Optional[int] = xconf.field(default=None, help="Height of search box")
    width : typing.Optional[int] = xconf.field(default=None, help="Width of search box")

@xconf.config
class BoxFromCenter(Box):
    center_x : int
    center_y : int

@xconf.config
class BoxFromOrigin(Box):
    origin_y : int
    origin_x : int

@xconf.config
class CutoutConfig:
    search_box : typing.Union[BoxFromCenter,BoxFromOrigin,Box] = xconf.field(default=Box(), help="Search box to find the PSF to cross-correlate with the template")
    template : typing.Union[base.FitsConfig, GaussianTemplate] = xconf.field(
        default=GaussianTemplate(),
        help=utils.unwrap("""
    Template cross-correlated with the search region to align images to a common grid, either given as a FITS image
    or specified as a centered 2D Gaussian with given FWHM
    """))
    use_first_as_template : bool = xconf.field(
        default=False,
        help="Use the template to cross correlate with the first frame, then correlate subsequent frames with that one"
    )


DEFAULT_CUTOUT = CutoutConfig(search_box=Box(), template=GaussianTemplate())


@xconf.config
class AlignedCutouts(base.MultiInputCommand):
    "Align PSF to template"
    cutouts : dict[str, CutoutConfig] = xconf.field(
        default_factory=lambda: {'cutout': DEFAULT_CUTOUT},
        help="Specify one or more cutouts with names and template PSFs to generate aligned cutouts for",
    )
    ext : typing.Union[str, int] = xconf.field(default=0, help="Extension index or name to load from input files")
    excluded_regions : typing.Optional[base.FileConfig] = xconf.field(default_factory=list, help="Regions to fill with zeros before cross-registration, stored as DS9 region file (reg format)")

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
        output_filepaths = [utils.join(self.destination, f"aligned_cutouts_{i:04}.fits") for i in range(n_output_files)]
        self.quit_if_outputs_exist(output_filepaths)

        input_coll = LazyPipelineCollection(all_inputs)
        coll = input_coll.map(iofits.load_fits_from_path)
        example_hdul = dask.compute(coll.items[0])[0]
        dimensions = example_hdul[self.ext].data.shape
        default_height, default_width = dimensions
        cutout_specs = []
        if isinstance(self.excluded_regions, base.FileConfig):
            with self.excluded_regions.open() as fh:
                excluded_regions = regions.load_file(fh)
            excluded_pixels_mask = regions.make_mask(excluded_regions, dimensions)
        else:
            excluded_pixels_mask = np.zeros(dimensions, dtype=bool)

        for name, cutout_config in self.cutouts.items():
            search_box = self._search_box_to_bbox(cutout_config.search_box, default_height, default_width)
            tpl = cutout_config.template
            if isinstance(tpl, GaussianTemplate):
                tpl_shape = tpl.size_px, tpl.size_px
                center = improc.arr_center(tpl_shape)
                template_array = improc.gauss2d(tpl_shape, center, (tpl.sigma_px, tpl.sigma_px))
            else:
                template_array = tpl.load()
            spec = improc.CutoutTemplateSpec(
                search_box=search_box,
                template=template_array,
                name=name
            )
            if cutout_config.use_first_as_template:
                data = example_hdul[self.ext].data
                data_driven_template = improc.aligned_cutout(data, spec)
                improc.interpolate_nonfinite(data_driven_template, data_driven_template)
                import numpy as np
                if np.any(np.isnan(data_driven_template)):
                    from astropy.io import fits
                    fits.PrimaryHDU(data_driven_template).writeto('./shifted_first_frame.fits', overwrite=True)
                spec.template = data_driven_template
            cutout_specs.append(spec)
            log.debug(spec)
        output_coll = pipelines.align_to_templates(coll, cutout_specs, excluded_pixels_mask)
        res = output_coll.zip_map(iofits.write_fits, output_filepaths, overwrite=True)

        result = res.compute()
        log.info(result)

    def _search_box_to_bbox(self, search_box, default_height, default_width):
        from ..tasks import improc
        height, width = search_box.height, search_box.width
        if height is None:
            height = default_height
        if width is None:
            width = default_width
        extent = improc.PixelExtent(width=width, height=height)
        if isinstance(search_box, BoxFromOrigin):
            yo, xo = search_box.origin_y, search_box.origin_x
            if xo is None:
                xo = 0
            if yo is None:
                yo = 0
            bbox = improc.BBox(
                origin=improc.Pixel(x=xo, y=yo),
                extent=extent
            )
        elif isinstance(search_box, BoxFromCenter):
            bbox = improc.BBox.from_center(center=improc.Pixel(y=search_box.center_y, x=search_box.center_x), extent=extent)
        else:
            bbox = improc.BBox(origin=improc.Pixel(y=0, x=0), extent=extent)
        return bbox
