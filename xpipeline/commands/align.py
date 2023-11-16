import sys
from xpipeline.core import LazyPipelineCollection, EagerPipelineCollection
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

    def to_bbox(self, shape):
        from ..tasks import improc
        extent = self.to_extent()
        return improc.BBox.from_center(center=improc.Pixel(y=self.center_y, x=self.center_x), extent=extent)

@xconf.config
class BoxFromOrigin(Box):
    origin_y : int
    origin_x : int

    def to_bbox(self, shape):
        from ..tasks import improc
        extent = self.to_extent()
        yo, xo = self.origin_y, self.origin_x
        return improc.BBox(
            origin=improc.Pixel(x=xo, y=yo),
            extent=extent
        )

@xconf.config
class GaussianTemplate:
    sigma_px : float = xconf.field(default=1, help="Template PSF kernel stddev in pixels")
    size_px : int = xconf.field(default=128, help="Size of generated Gaussian template in pixels")

    def load(self):
        from ..tasks import improc
        tpl_shape = self.size_px, self.size_px
        center = improc.arr_center(tpl_shape)
        template_array = improc.gauss2d(tpl_shape, center, (self.sigma_px, self.sigma_px))
        return template_array


@xconf.config
class FeatureConfig:
    search_box : typing.Union[BoxFromCenter,BoxFromOrigin,None] = xconf.field(default=None, help="Search box to find the PSF to cross-correlate with the template, default uses whole image")
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
    faux_saturation_percentile : float = xconf.field(default=100, help="Percentile at which to clip template to simulate saturation")


DEFAULT_CENTER_FEATURE = FeatureConfig()

def fit_feature(masked_data, spec):
    from ..tasks import improc
    center_point = improc.subpixel_location_from_cutout(masked_data, spec)
    return center_point

def feature_coords_to_offsets(feature_point, true_center):
    from ..tasks import improc
    print(f"{feature_point=} {true_center=}")
    return improc.Point(y=true_center.y - feature_point.y, x=true_center.x - feature_point.x)


def mean_combine_points(point_collection):
    import numpy as np
    from ..tasks import improc
    xs, ys = [], []
    for p in point_collection:
        xs.append(p.x)
        ys.append(p.y)
    return improc.Point(x=np.mean(xs), y=np.mean(ys))

def estimate_true_center(sci_arr, *offsets):
    import numpy as np
    from ..tasks import improc
    xs = [o.x for o in offsets]
    ys = [o.y for o in offsets]
    return improc.Point(y=np.mean(ys), x=np.mean(xs))

def feature_location_to_center(feature, offset):
    from ..tasks import improc
    return improc.Point(y=feature.y + offset.y, x=feature.x + offset.x)

def make_aligned_image(sci_arr, true_center, new_dimensions, new_ctr):
    from ..tasks import improc
    dx, dy = new_ctr.x - true_center.x, new_ctr.y - true_center.y
    new_shape = (new_dimensions.height, new_dimensions.width)
    return improc.shift2(sci_arr.astype(float), dx, dy, new_shape)

def construct_output_fits(sci_arr, hdul, ext):
    new_hdul = hdul.copy()
    new_hdul[ext].data = sci_arr
    return new_hdul

@xconf.config
class Align(base.MultiInputCommand):
    "Align images to common center"
    saturated_input : typing.Union[str, None] = xconf.field(default=None, help="If applicable, saturated frames to reference to these unsaturated input frames")
    center : FeatureConfig = xconf.field(default=DEFAULT_CENTER_FEATURE, help="Configuration for the search region for the rotation center of these input images and centered template to use")
    registration_features : typing.Optional[list[typing.Union[BoxFromCenter,BoxFromOrigin]]] = xconf.field(default_factory=list, help="Regions containing unsaturated features to correlate on")
    excluded_regions : typing.Optional[base.FileConfig] = xconf.field(default=None, help="Regions to fill with zeros before cross-registration, stored as DS9 region file (reg format)")
    crop_to : float = xconf.field(default=1.0, help="Crop to this value times the original dimensions when shifting (make this >1.0 if this shifts are moving regions of interest out of view)")
    ext : typing.Union[str, int] = xconf.field(default=0, help="Extension index or name to load from input files")
    dq_ext : typing.Union[str, int] = xconf.field(default="DQ", help="Extension index or name for data quality array")
    ray : base.AnyRayConfig = xconf.field(
        default=base.LocalRayConfig(),
        help="Ray distributed framework configuration"
    )


    def main(self):
        import dask
        from ray.util.dask import ray_dask_get
        from .. import pipelines
        from ..tasks import iofits, improc, regions, data_quality
        from .. import constants as const
        import numpy as np

        self.ray.init()

        log.debug(self)
        dest_fs = utils.get_fs(self.destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {self.destination}")
        dest_fs.makedirs(self.destination, exist_ok=True)

        all_inputs = self.get_all_inputs(self.input)
        n_output_files = len(all_inputs)
        output_filepaths = [utils.join(self.destination, f"aligned_{i:04}.fits") for i in range(n_output_files)]
        all_output_filepaths = output_filepaths.copy()

        if self.saturated_input is not None:
            sat_inputs = self.get_all_inputs(self.saturated_input)
            output_sat_filepaths = [utils.join(self.destination, f"saturated_aligned_{i:04}.fits") for i in range(len(sat_inputs))]
            all_output_filepaths.extend(output_sat_filepaths)
        self.quit_if_outputs_exist(all_output_filepaths)

        input_coll = LazyPipelineCollection(all_inputs)
        # input_coll = EagerPipelineCollection(all_inputs)
        coll = input_coll.map(iofits.load_fits_from_path).precompute(scheduler=ray_dask_get)
        example_hdul = dask.compute(coll.items[0])[0]
        # example_hdul = coll.items[0]
        example_data = example_hdul[self.ext].data
        ext = self.ext
        dq_ext = self.dq_ext
        original_dimensions = example_data.shape
        print(f"{self.center.search_box=}")
        if self.center.search_box is None:
            self.center.search_box = BoxFromOrigin(height=original_dimensions[0], width=original_dimensions[1], origin_x=0, origin_y=0)
        print(f"{self.center.search_box=}")
        new_dimensions = improc.PixelExtent(height=int(self.crop_to * example_data.shape[0]), width=int(self.crop_to * example_data.shape[1]))
        new_ctr = improc.center_point(new_dimensions)
        log.debug(f"Using {self.crop_to} crop factor, shifted frames have shape {new_dimensions} and center {new_ctr}")

        if isinstance(self.excluded_regions, base.FileConfig):
            with self.excluded_regions.open() as fh:
                excluded_regions = regions.load_file(fh)
            excluded_pixels_mask = regions.make_mask(excluded_regions, original_dimensions)
        else:
            excluded_pixels_mask = np.zeros(original_dimensions, dtype=bool)
        
        # fit true centers
        masked_data = coll.map(data_quality.get_masked_data, ext=ext, dq_ext=dq_ext, permitted_flags=const.DQ_SATURATED, excluded_pixels_mask=excluded_pixels_mask)
        print(f"{self.center=}")
        true_centers = (masked_data
            .map(fit_feature, improc.ImageFeatureSpec(self.center.search_box.to_bbox(original_dimensions), self.center.template.load()))
            .end()
        )
        aligned_frame_paths = (
            masked_data
            .zip_map(make_aligned_image, true_centers, new_dimensions, new_ctr)
            .zip_map(construct_output_fits, coll.items, self.ext)
            .zip_map(iofits.write_fits, output_filepaths)
            .end()
        )
        
        # measure deltas from true centers to reference features
        d_offsets_per_feature = []
        for feature_spec in self.registration_features:
            search_box = feature_spec.to_bbox(original_dimensions)
            print(f"{search_box=}")
            initial_template = example_data[search_box.slices]
            d_offsets_per_feature.append((masked_data
                .map(fit_feature, improc.ImageFeatureSpec(search_box=search_box, template=initial_template))
                .zip_map(feature_coords_to_offsets, true_centers)
                .collect(mean_combine_points)
            ))
        
        # load saturated frames
        if self.saturated_input is not None:
            sat_hduls = LazyPipelineCollection(sat_inputs).map(iofits.load_fits_from_path)
            sat_masked_data = (
                sat_hduls
                .map(data_quality.get_masked_data, ext=ext, dq_ext=dq_ext, permitted_flags=const.DQ_SATURATED, excluded_pixels_mask=excluded_pixels_mask)
            )
            d_sat_frame_centers_by_feature = []
            for feature_spec, d_center_offset in zip(self.registration_features, d_offsets_per_feature):
                search_box = feature_spec.to_bbox(original_dimensions)
                print(f"{search_box=}")
                initial_template = example_data[search_box.slices]
                d_sat_frame_centers_by_feature.append((sat_masked_data
                    # fit features in sat frames
                    .map(fit_feature, improc.ImageFeatureSpec(search_box=search_box, template=initial_template))
                    # apply delta from feature to center to each feature measurement to get center estimate
                    .map(feature_location_to_center, d_center_offset)
                ).items)
            # average per-feature center estimates to get a frame center estimate
            d_true_centers = sat_masked_data.zip_map(estimate_true_center, *d_sat_frame_centers_by_feature).items
            aligned_sat_frame_paths = (sat_masked_data
                .zip_map(make_aligned_image, d_true_centers, new_dimensions, new_ctr)
                .zip_map(construct_output_fits, coll.items, self.ext)
                .zip_map(iofits.write_fits, output_sat_filepaths)
                .end()
            )
            # print(aligned_sat_frame_paths.compute(scheduler=ray_dask_get))
        else:
            aligned_sat_frame_paths = tuple()

        print(dask.compute([aligned_frame_paths, aligned_sat_frame_paths], scheduler=ray_dask_get))