import numpy as np
import pathlib
import tqdm
import sys
from xpipeline.core import LazyPipelineCollection, EagerPipelineCollection
import fsspec.spec
from dataclasses import dataclass
import logging
import typing
import ray
import xconf
from typing import Union, Optional
from astropy.io import fits
from numpy.typing import ArrayLike
from .. import utils, constants
from ..tasks import improc, iofits
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
        default_factory=GaussianTemplate,
        help=utils.unwrap("""
    Template cross-correlated with the search region to align images to a common grid, either given as a FITS image
    or specified as a centered 2D Gaussian with given FWHM
    """))
    use_first_as_template : bool = xconf.field(
        default=False,
        help="Use the template to cross correlate with the first frame, then correlate subsequent frames with that one"
    )
    faux_saturation_percentile : float = xconf.field(default=100, help="Percentile at which to clip template to simulate saturation")

@xconf.config
class RegistrationFeature:
    ref_box : typing.Union[BoxFromCenter,BoxFromOrigin] = xconf.field(help="Coordinates of feature in reference images")
    search_box : typing.Union[BoxFromCenter,BoxFromOrigin] = xconf.field(help="Box containing feature in obscured-peak images")

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
    return improc.shift2(sci_arr.astype(float), dx, dy, new_shape, anchor_to_center=True)

def construct_output_fits(sci_arr, hdul, ext):
    new_hdul = hdul.copy()
    new_hdul[ext].data = sci_arr
    return new_hdul

def _load_fits_frame(frame_fn) -> iofits.PicklableHDUList:
    return iofits.load_fits_from_path(frame_fn)
load_fits_frame = ray.remote(_load_fits_frame)


def _write_fits_frame(frame_hdul: iofits.PicklableHDUList, shifted_frame: ArrayLike, obsc_output_fn: pathlib.Path, ext : Union[int, str]=0):
    shifted_frame_hdul = frame_hdul.updated_copy(new_data_for_exts={ext: shifted_frame})
    iofits.write_fits(shifted_frame_hdul, obsc_output_fn)
    return obsc_output_fn
write_fits_frame = ray.remote(_write_fits_frame)

def _data_from_hdul(hdul: fits.HDUList, ext: Union[str,int]):
    return hdul[ext].data.astype('=f8')
data_from_hdul = ray.remote(_data_from_hdul)

@dataclass
class RegistrationMeasurements:
    features: list[improc.Point]
    center: Optional[improc.Point] = None

def _measure_features(data: ArrayLike, specs: list[RegistrationFeature], center_spec: Optional[RegistrationFeature]=None, usm_sigma_px: Optional[float]=None) -> RegistrationMeasurements:
    from ..tasks import improc
    if usm_sigma_px is not None:
        data = data - improc.gaussian_smooth(data, usm_sigma_px)
    feature_measurements: list[improc.Point] = []
    for spec in specs:
        feature_measurements.append(improc.subpixel_location_from_cutout(data, spec))
    center_measurement = None
    if center_spec is not None:
        center_measurement = improc.subpixel_location_from_cutout(data, center_spec)
    return RegistrationMeasurements(features=feature_measurements, center=center_measurement)
measure_features = ray.remote(_measure_features)

def _make_shifted_frame(frame: ArrayLike, frame_ctr: improc.Point, new_ctr: improc.Point, output_extent : improc.PixelExtent):
    dx, dy = new_ctr.x - frame_ctr.x, new_ctr.y - frame_ctr.y
    new_shape = (output_extent.height, output_extent.width)
    return improc.shift2(frame, dx, dy, new_shape, anchor_to_center=False)
make_shifted_frame = ray.remote(_make_shifted_frame)

def _summarize_offsets(*offsets: RegistrationMeasurements) -> list[improc.DeltaPoint]:
    # note mildly cursed varargs is so futures get awaited correctly
    assert len(offsets) > 0
    n_features = len(offsets[0].features)
    # lists to hold (dxs, dys)
    per_feature_deltas = [([], []) for _ in range(n_features)]
    for meas in offsets:
        for idx, feat_meas in enumerate(meas.features):
            dx = meas.center.x - feat_meas.x
            dy = meas.center.y - feat_meas.y
            per_feature_deltas[idx][0].append(dx)
            per_feature_deltas[idx][1].append(dy)
    log.info(f"Using {len(offsets)} measurements of {n_features} features to get feature-to-center offsets")
    median_deltas = []
    for i in range(n_features):
        mean_dx = np.average(per_feature_deltas[i][0])
        mean_dy = np.average(per_feature_deltas[i][1])
        median_deltas.append(improc.DeltaPoint(dy=mean_dy, dx=mean_dx))
    return median_deltas
summarize_offsets = ray.remote(_summarize_offsets)

def _center_from_features(feature_measurements: RegistrationMeasurements, feature_offsets: list[improc.DeltaPoint]) -> improc.Point:
    center_est_xs = []
    center_est_ys = []
    for meas, offst in zip(feature_measurements.features, feature_offsets):
        center_est_xs.append(meas.x + offst.dx)
        center_est_ys.append(meas.y + offst.dy)
    # TODO track uncertainties
    return improc.Point(x=np.average(center_est_xs), y=np.average(center_est_ys))
center_from_features = ray.remote(_center_from_features)

def _measure_and_shift_ref_frame(
        frame_fn : str,
        output_fn: str,
        ext : Union[int,str],
        feature_specs: list[improc.ImageFeatureSpec],
        center_spec: improc.ImageFeatureSpec,
        new_ctr : improc.Point,
        new_dimensions: improc.PixelExtent,
        usm_sigma_px: float,
):
    frame_hdul = _load_fits_frame(frame_fn)
    frame_data = _data_from_hdul(frame_hdul, ext=ext)

    # collect reg. feature-to-center offsets and center coords
    measurements = _measure_features(
        frame_data,
        feature_specs,
        center_spec=center_spec,
        usm_sigma_px=usm_sigma_px
    )
    if output_fn is not None:
        shifted_frame = _make_shifted_frame(
            frame_data,
            measurements.center,
            new_ctr,
            new_dimensions
        )
        _write_fits_frame(frame_hdul, shifted_frame, output_fn, ext=ext)
    return measurements
measure_and_shift_ref_frame = ray.remote(_measure_and_shift_ref_frame)

def _measure_and_shift_obsc_frame(
        obsc_frame_fn : str,
        obsc_output_fn: str,
        ext : Union[int,str],
        feature_specs: list[improc.ImageFeatureSpec],
        new_ctr : improc.Point,
        new_dimensions: improc.PixelExtent,
        usm_sigma_px: float,
        offsets_summary: list[improc.DeltaPoint],
):
    frame_hdul = _load_fits_frame(obsc_frame_fn)
    frame_data = _data_from_hdul(frame_hdul, ext=ext)
    measure_features_ref = _measure_features(frame_data, feature_specs, usm_sigma_px=usm_sigma_px)
    center_est = _center_from_features(measure_features_ref, offsets_summary)
    shifted_frame = _make_shifted_frame(frame_data, center_est, new_ctr, new_dimensions)
    return _write_fits_frame(frame_hdul, shifted_frame, obsc_output_fn, ext=ext)
measure_and_shift_obsc_frame = ray.remote(_measure_and_shift_obsc_frame)


@xconf.config
class Align(base.MultiInputCommand):
    "Align images to common center"
    reference_frame : typing.Optional[base.FitsImageConfig] = xconf.field(default=None, help="Use this instead of the first frame to measure reference feature locations")
    obscured_input : typing.Union[str, None] = xconf.field(default=None, help="If applicable, saturated frames to reference to these unsaturated input frames")
    center : FeatureConfig = xconf.field(default_factory=FeatureConfig, help="Configuration for the search region for the rotation center of these input images and centered template to use")
    registration_features : list[typing.Union[RegistrationFeature,BoxFromCenter,BoxFromOrigin]] = xconf.field(default_factory=list, help="Regions containing unsaturated features to correlate on")
    # excluded_regions : typing.Optional[base.FileConfig] = xconf.field(default=None, help="Regions to fill with zeros before cross-registration, stored as DS9 region file (reg format)")
    crop_to : float = xconf.field(default=1.0, help="Crop to this value times the original dimensions when shifting (make this >1.0 if this shifts are moving regions of interest out of view)")
    ext : typing.Union[str, int] = xconf.field(default=0, help="Extension index or name to load from input files")
    dq_ext : typing.Union[str, int] = xconf.field(default="DQ", help="Extension index or name for data quality array")
    ray : base.AnyRayConfig = xconf.field(
        default_factory=base.LocalRayConfig,
        help="Ray distributed framework configuration"
    )
    # prefilter_sigma_px : float = xconf.field(default=0.0, help=">0 values mean Gaussian smoothing of the images before trying to register features")
    usm_sigma_px : float = xconf.field(default=0.0, help=">0 values indicate the smoothing kernel width for an unsharp mask before trying to register features")
    batch_size : int = xconf.field(default=16)

    def main(self):
        from ..tasks import iofits, improc, regions, data_quality
        from .. import constants as const
        import numpy as np

        self.ray.init()

        log.debug(self)
        dest_fs = utils.get_fs(self.destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {self.destination}")
        dest_fs.makedirs(self.destination, exist_ok=True)

        all_ref_inputs = self.get_all_inputs(self.input)
        n_output_files = len(all_ref_inputs)
        output_filepaths = [utils.join(self.destination, f"aligned_{i:06}.fits") for i in range(n_output_files)]
        all_output_filepaths = output_filepaths.copy()

        obsc_inputs = []
        output_obsc_filepaths = []
        if self.obscured_input is not None:
            obsc_inputs = self.get_all_inputs(self.obscured_input)
            output_obsc_filepaths = [utils.join(self.destination, f"obscured_aligned_{i:06}.fits") for i in range(len(obsc_inputs))]
            all_output_filepaths.extend(output_obsc_filepaths)
        self.quit_if_outputs_exist(all_output_filepaths)

        if self.reference_frame is None:
            self.reference_frame = base.FitsImageConfig(path=all_ref_inputs[0], ext=self.ext)
        reference_frame_data = self.reference_frame.load()
        original_dimensions = reference_frame_data.shape

        # If no search box specified, default to full frame
        if self.center.search_box is None:
            self.center.search_box = BoxFromOrigin(height=original_dimensions[0], width=original_dimensions[1], origin_x=0, origin_y=0)

        # Apply crop, calculate new dimensions / center
        new_dimensions = improc.PixelExtent(height=int(self.crop_to * reference_frame_data.shape[0]), width=int(self.crop_to * reference_frame_data.shape[1]))
        new_ctr = improc.center_point(new_dimensions)
        orig_geom_ctr = improc.center_point(reference_frame_data)
        log.debug(f"Using {self.crop_to} crop factor, shifted frames have shape {new_dimensions} and center {new_ctr}")

        # construct ImageFeatureSpecs and templates
        feature_specs = []
        center_spec = improc.ImageFeatureSpec(
            search_box=self.center.search_box.to_bbox(original_dimensions),
            template=self.center.template.load()
        )
        for feature_config in self.registration_features:
            if isinstance(feature_config, RegistrationFeature):
                search_box = feature_config.ref_box.to_bbox(original_dimensions)
            else:
                search_box = feature_config.to_bbox(original_dimensions)
            log.debug(f"make {search_box=} search")
            initial_template = reference_frame_data[search_box.slices]
            feature_specs.append(improc.ImageFeatureSpec(search_box=search_box, template=initial_template))

        # submit futures for fit ref centers and reg features
        all_feature_measurements = []
        shifted_output_refs = []
        
        ram_requirement_mb, _ = utils.measure_ray_task_memory(
            _measure_and_shift_ref_frame,
            {},
            args=(all_ref_inputs[0], None, self.ext, feature_specs, center_spec, new_ctr, new_dimensions, self.usm_sigma_px)
        )
        log.info(f"{ram_requirement_mb=}")
        task_options = {'memory': ram_requirement_mb * 1024 * 1024}  # bytes

        all_feature_measurements = []
        for frame_fn, output_fn in zip(all_ref_inputs, output_filepaths):
            # _measure_and_shift_ref_frame(frame_fn, output_fn, self.ext, feature_specs, center_spec, new_ctr, new_dimensions, self.usm_sigma_px)
            ref_measurement = measure_and_shift_ref_frame.options(**task_options).remote(frame_fn, output_fn, self.ext, feature_specs, center_spec, new_ctr, new_dimensions, self.usm_sigma_px)
            all_feature_measurements.append(ref_measurement)

        # submit offset summary future with all fit results as pending inputs
        # (separate args so the objectrefs get resolved before call)
        offsets_summary_ref = summarize_offsets.remote(*all_feature_measurements)

        # chunk through shifted output tasks
        with tqdm.tqdm(desc='ref frames', total=len(output_filepaths)) as pbar:
            while shifted_output_refs:
                finished_output_refs, shifted_output_refs = ray.wait(shifted_output_refs, num_returns=min(self.batch_size, len(shifted_output_refs)))
                ray.get(finished_output_refs)  # in case any exceptions need to be re-raised
                pbar.update(len(finished_output_refs))
        assert len(shifted_output_refs) == 0

        # retrieve the summary
        offsets_summary = ray.get(offsets_summary_ref)
        log.info(offsets_summary)

        if self.obscured_input is None:
            return

        obsc_outputs = []
        for obsc_frame_fn, obsc_output_fn in zip(obsc_inputs, output_obsc_filepaths):
            ref = measure_and_shift_obsc_frame.options(**task_options).remote(
                obsc_frame_fn,
                obsc_output_fn,
                self.ext,
                feature_specs,
                new_ctr,
                new_dimensions,
                self.usm_sigma_px,
                offsets_summary
            )
            obsc_outputs.append(ref)

        # chunk through shifted output tasks
        n_obsc = len(output_obsc_filepaths)
        with tqdm.tqdm(desc='obsc frames', total=n_obsc) as pbar:
            while obsc_outputs:
                finished_output_refs, obsc_outputs = ray.wait(obsc_outputs, num_returns=min(self.batch_size, len(obsc_outputs)))
                ray.get(finished_output_refs)  # in case any exceptions need to be re-raised
                pbar.update(len(finished_output_refs))

        return
        ## old

        input_coll = LazyPipelineCollection(all_ref_inputs)
        # input_coll = EagerPipelineCollection(all_inputs)
        coll = input_coll.map(iofits.load_fits_from_path).precompute(scheduler=ray_dask_get)
        reference_hdul = dask.compute(coll.items[0])[0]
        # example_hdul = coll.items[0]
        if self.center.search_box is None:
            reference_frame_data = reference_hdul[self.ext].data
            ext = self.ext
            dq_ext = self.dq_ext
            original_dimensions = reference_frame_data.shape
            self.center.search_box = BoxFromOrigin(height=original_dimensions[0], width=original_dimensions[1], origin_x=0, origin_y=0)
        new_dimensions = improc.PixelExtent(height=int(self.crop_to * reference_frame_data.shape[0]), width=int(self.crop_to * reference_frame_data.shape[1]))
        new_ctr = improc.center_point(new_dimensions)
        orig_geom_ctr = improc.center_point(reference_frame_data)
        log.debug(f"Using {self.crop_to} crop factor, shifted frames have shape {new_dimensions} and center {new_ctr}")

        if isinstance(self.excluded_regions, base.FileConfig):
            with self.excluded_regions.open() as fh:
                excluded_regions = regions.load_file(fh)
            excluded_pixels_mask = regions.make_mask(excluded_regions, original_dimensions)
        else:
            excluded_pixels_mask = np.zeros(original_dimensions, dtype=bool)

        # fit true centers
        masked_data = coll.map(data_quality.get_masked_data, ext=ext, dq_ext=dq_ext, permitted_flags=const.DQ_SATURATED, excluded_pixels_mask=excluded_pixels_mask)
        registration_data = masked_data
        print(f"{self.center=}")
        if self.prefilter_sigma_px > 0:
            registration_data = registration_data.map(lambda x: improc.gaussian_smooth(x, self.prefilter_sigma_px))
        if self.usm_sigma_px > 0:
            registration_data = registration_data.map(lambda x: x - improc.gaussian_smooth(x, self.usm_sigma_px))
        true_centers = (registration_data
            .map(fit_feature, improc.ImageFeatureSpec(self.center.search_box.to_bbox(original_dimensions), self.center.template.load()))
            .end()
        )
        aligned_images = masked_data.zip_map(make_aligned_image, true_centers, new_dimensions, orig_geom_ctr).precompute()
        aligned_frame_paths = (
            aligned_images
            .zip_map(construct_output_fits, coll.items, self.ext)
            .zip_map(iofits.write_fits, output_filepaths)
            .end()
        )

        # measure deltas from true centers to reference features
        d_offsets_per_feature = []
        for feature_spec in self.registration_features:
            if isinstance(feature_spec, RegistrationFeature):
                search_box = feature_spec.ref_box.to_bbox(original_dimensions)
            else:
                search_box = feature_spec.to_bbox(original_dimensions)
            log.debug(f"make {search_box=} search")
            initial_template = reference_frame_data[search_box.slices]
            d_offsets_per_feature.append((registration_data
                .map(fit_feature, improc.ImageFeatureSpec(search_box=search_box, template=initial_template))
                .zip_map(feature_coords_to_offsets, true_centers)
                .collect(mean_combine_points)
            ))

        # load saturated frames
        if self.obscured_input is not None:
            d_median_unsat = masked_data.collect(improc.combine, constants.CombineOperation.MEDIAN)
            median_unsat, = dask.compute(d_median_unsat)
            sat_hduls = LazyPipelineCollection(obsc_inputs).map(iofits.load_fits_from_path).precompute(scheduler=ray_dask_get)
            sat_masked_data = (
                sat_hduls
                .map(data_quality.get_masked_data, ext=ext, dq_ext=dq_ext, permitted_flags=const.DQ_SATURATED, excluded_pixels_mask=excluded_pixels_mask)
            )
            obscured_registration_data = sat_masked_data
            if self.prefilter_sigma_px > 0:
                obscured_registration_data = obscured_registration_data.map(lambda x: improc.gaussian_smooth(x, self.prefilter_sigma_px))
            if self.usm_sigma_px > 0:
                obscured_registration_data = obscured_registration_data.map(lambda x: x - improc.gaussian_smooth(x, self.usm_sigma_px))


            d_sat_frame_centers_by_feature = []
            for feature_spec, d_center_offset in zip(self.registration_features, d_offsets_per_feature):
                if isinstance(feature_spec, RegistrationFeature):
                    ref_search_box = feature_spec.ref_box.to_bbox(original_dimensions)
                    search_box = feature_spec.search_box.to_bbox(original_dimensions)
                else:
                    ref_search_box = search_box = feature_spec.to_bbox(original_dimensions)
                log.debug(f"make {search_box=} search in saturated")
                initial_template = median_unsat[ref_search_box.slices]
                d_sat_frame_centers_by_feature.append((obscured_registration_data
                    # fit features in sat frames
                    .map(fit_feature, improc.ImageFeatureSpec(search_box=search_box, template=initial_template))
                    # apply delta from feature to center to each feature measurement to get center estimate
                    .map(feature_location_to_center, d_center_offset)
                ).items)
            # average per-feature center estimates to get a frame center estimate
            d_true_centers = sat_masked_data.zip_map(estimate_true_center, *d_sat_frame_centers_by_feature).items
            aligned_sat_frame_paths = (sat_masked_data
                .zip_map(make_aligned_image, d_true_centers, new_dimensions, orig_geom_ctr)
                .zip_map(construct_output_fits, sat_hduls.items, self.ext)
                .zip_map(iofits.write_fits, output_obsc_filepaths)
                .end()
            )
            # print(aligned_sat_frame_paths.compute(scheduler=ray_dask_get))
        else:
            aligned_sat_frame_paths = tuple()

        print(dask.compute([aligned_frame_paths, aligned_sat_frame_paths], scheduler=ray_dask_get))
