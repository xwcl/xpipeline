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
class FileTemplate:
    path : str = xconf.field(help="Path to template FITS image")
    ext : typing.Union[int, str] = xconf.field(default=0, help="Extension containing image data")


@xconf.config
class GaussianTemplate:
    sigma_px : float = xconf.field(default=10, help="Template PSF kernel stddev in pixels")
    size_px : int = xconf.field(default=128, help="Size of generated Gaussian template in pixels")


@xconf.config
class CutoutConfig:
    search_box_origin_y : typing.Optional[int]
    search_box_origin_x : typing.Optional[int]
    search_box_height : typing.Optional[int]
    search_box_width : typing.Optional[int]
    template : typing.Union[FileTemplate, GaussianTemplate] = xconf.field(
        default=GaussianTemplate(),
        help=utils.unwrap("""
    Template cross-correlated with the search region to align images to a common grid, either given as a FITS image
    or specified as a centered 2D Gaussian with given FWHM
    """))


DEFAULT_CUTOUT = CutoutConfig(search_box_origin_y=None, search_box_origin_x=None, search_box_height=None,
                              search_box_width=None, template=GaussianTemplate())


@xconf.config
class AlignedCutouts(base.MultiInputCommand):
    "Align PSF to template"
    cutouts : dict[str, CutoutConfig] = xconf.field(
        default_factory=lambda: {'cutout': DEFAULT_CUTOUT},
        help="Specify one or more cutouts with names and template PSFs to generate aligned cutouts for",
    )
    ext : typing.Union[str, int] = xconf.field(default=0, help="Extension index or name to load from input files")

    def main(self):
        import dask
        from .. import pipelines
        from ..tasks import iofits, improc

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
        for name, cutout_config in self.cutouts.items():
            yo, xo = cutout_config.search_box_origin_y, cutout_config.search_box_origin_x
            if xo is None:
                xo = 0
            if yo is None:
                yo = 0
            height, width = cutout_config.search_box_height, cutout_config.search_box_width
            if height is None:
                height = default_height
            if width is None:
                width = default_width
            search_box = improc.BBox(
                origin=improc.Pixel(x=xo, y=yo),
                extent=improc.PixelExtent(width=width, height=height)
            )
            tpl = cutout_config.template
            if isinstance(tpl, GaussianTemplate):
                tpl_shape = tpl.size_px, tpl.size_px
                center = improc.arr_center(tpl_shape)
                template_array = improc.gauss2d(tpl_shape, center, (tpl.sigma_px, tpl.sigma_px))
            else:
                hdul = iofits.load_fits_from_path()
                template_array = hdul[tpl.ext].data
            spec = improc.CutoutTemplateSpec(
                search_box=search_box,
                template=template_array,
                name=name
            )
            cutout_specs.append(spec)
            log.debug(spec)
        output_coll = pipelines.align_to_templates(coll, cutout_specs)
        res = output_coll.zip_map(iofits.write_fits, output_filepaths, overwrite=True)

        result = res.compute()
        log.info(result)
