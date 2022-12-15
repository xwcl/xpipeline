import logging
from typing import Optional, Union
import xconf
import pandas as pd
from xpipeline.ref import clio, magellan
from xpipeline.tasks import characterization
from .. import utils, constants, types
from .base import FitsConfig, InputCommand
import astropy.units as u

log = logging.getLogger(__name__)

import numpy as np

@xconf.config
class GridColumnsConfig:
    r_px : str = xconf.field(default="r_px", help="Radius (separation) in pixels")
    pa_deg : str = xconf.field(default="pa_deg", help="PA (angle) in degrees E of N (+Y)")
    snr : str = xconf.field(default="snr", help="Signal-to-noise ratio for a given location and parameters")
    injected_scale : str = xconf.field(default="injected_scale", help="(companion)/(host) contrast of injection")
    hyperparameters : list[str] = xconf.field(default_factory=lambda: ['k_modes'], help="List of columns holding hyperparameters varied in grid")

FilterValue = Union[float, str]

@xconf.config
class FilterPredicate:
    operator : constants.CompareOperation = xconf.field(help="Any of gt/ge/eq/le/lt/ne")
    value : FilterValue = xconf.field(help="Value on which to filter")

FilterSpec = Union[FilterPredicate, float, str]

@xconf.config
class SummarizeGrid(InputCommand):
    ext : str = xconf.field(default="grid", help="FITS binary table extension with calibration grid")
    columns : GridColumnsConfig = xconf.field(default=GridColumnsConfig(), help="Grid column names")
    arcsec_per_pixel : float = xconf.field(default=clio.CLIO2_PIXEL_SCALE.value)
    wavelength_um : float = xconf.field(help="Wavelength in microns")
    primary_diameter_m : float = xconf.field(default=magellan.PRIMARY_MIRROR_DIAMETER.to(u.m).value)
    coverage_mask : FitsConfig = xconf.field(help="Mask image with 1s where pixels have observation coverage and 0 elsewhere")
    min_coverage : float = xconf.field(default=1.0, help="minimum number of frames covering a pixel for it to be used in the final interpolated map")
    min_snr_for_injection: float = xconf.field(default=10, help="Minimum SNR to recover in order to trust the SNR=5 contrast value")
    non_detection_threshold: float = xconf.field(default=3, help="Minimum SNR in detection map to disqualify a calibration point's limiting contrast value")
    normalize_snrs: Union[bool, list[str]] = xconf.field(default=False, help="Rescales the SNR values at a given radius by dividing by the stddev of all SNRs for that radius, if given as list of str, is used as names of grouping params")
    filters : dict[str, FilterSpec] = xconf.field(default_factory=dict, help="Filter on column == value before grouping and summarizing")
    enforce_radial_spacing : bool = xconf.field(default=True, help="Ensures radii are >= 1 lambda/D apart")

    def main(self):
        import pandas as pd
        from ..tasks import iofits, improc
        output_filepath, = self.get_output_paths("summarize_grid.fits")
        self.quit_if_outputs_exist([output_filepath])

        coverage_mask = self.coverage_mask.load() >= self.min_coverage
        yc, xc = improc.arr_center(coverage_mask)

        grid_hdul = iofits.load_fits_from_path(self.input)
        grid_tbl = grid_hdul[self.ext].data
        log.info(f"Loaded {len(grid_tbl)} points for evaluation")

        grid_df = pd.DataFrame(grid_tbl)
        mask = np.ones(len(grid_df), dtype=bool)
        for column_key in self.filters:
            this_filter = self.filters[column_key]
            this_column = grid_df[column_key]
            if isinstance(this_filter, FilterPredicate):
                op = this_filter.operator
                val = this_filter.value
                log.debug(f"Filtering on {column_key} {op.ascii_operator} {val}")
                if op is constants.CompareOperation.EQ:
                    mask &= (this_column == val)
                elif op is constants.CompareOperation.NE:
                    mask &= (this_column != val)
                elif op is constants.CompareOperation.GT:
                    mask &= (this_column > val)
                elif op is constants.CompareOperation.GE:
                    mask &= (this_column >= val)
                elif op is constants.CompareOperation.LT:
                    mask &= (this_column < val)
                elif op is constants.CompareOperation.LE:
                    mask &= (this_column <= val)
            else:
                log.debug(f"Filtering on {column_key} == {this_filter}")
                if isinstance(this_filter, str):
                    this_filter = this_filter.encode('utf8')
                mask &= (grid_df[column_key] == this_filter)
            log.debug(f"Remaining rows = {np.count_nonzero(mask)}")

        grid_df = grid_df[mask]
        grid_df = grid_df[~pd.isna(grid_df[self.columns.snr])]

        if self.normalize_snrs:
            log.debug("Applying SNR rescaling by azimuthal stddev of non-injected SNR measurements")
            if isinstance(self.normalize_snrs, list):
                extra_grouping_colnames = self.normalize_snrs
            else:
                extra_grouping_colnames = self.columns.hyperparameters
            grid_tbl = characterization.normalize_snr_for_grid(
                grid_tbl,
                group_by_colname=self.columns.r_px,
                snr_colname=self.columns.snr,
                injected_scale_colname=self.columns.injected_scale,
                hyperparameter_colnames=extra_grouping_colnames,
            )

        limits_df, detections_df = characterization.summarize_grid(
            grid_df,
            r_px_colname=self.columns.r_px,
            pa_deg_colname=self.columns.pa_deg,
            snr_colname=self.columns.snr,
            injected_scale_colname=self.columns.injected_scale,
            hyperparameter_colnames=self.columns.hyperparameters,
            min_snr_for_injection=self.min_snr_for_injection,
            non_detection_threshold=self.non_detection_threshold,
        )
        limits_df['delta_mag_contrast_limit_5sigma'] = characterization.contrast_to_deltamag(limits_df['contrast_limit_5sigma'].to_numpy())

        if self.enforce_radial_spacing:
            lambda_over_d = characterization.lambda_over_d_to_arcsec(1, self.wavelength_um * u.um, self.primary_diameter_m * u.m)
            radial_spacing_px = (lambda_over_d / (self.arcsec_per_pixel * u.arcsec / u.pixel)).to(u.pixel).value
            radii = np.unique(grid_df[self.columns.r_px])
            spaced_radii = []
            last_radius = 0
            for radius in radii:
                # we check if *any* limits were calibrated successfully at this radius
                # so we can skip radii where there's only garbage
                if np.count_nonzero(limits_df[self.columns.r_px] == radius) and radius - last_radius >= radial_spacing_px:
                    last_radius = radius
                    spaced_radii.append(radius)
            log.debug(f"Keeping radii spaced by {radial_spacing_px} px gives these: {spaced_radii}")

            detections_mask = np.zeros(len(detections_df), dtype=bool)
            for radius in spaced_radii:
                detections_mask |= (detections_df[self.columns.r_px]).to_numpy() == radius

            log.debug(f"Before radial spacing enforcement (>= {radial_spacing_px:3.1f} px): {len(detections_df)=}")
            detections_df = detections_df[detections_mask].copy()
            log.debug(f"Keeping the full {len(limits_df)} rows in limits, but {len(detections_df)} rows in detections")

        for df in [limits_df, detections_df]:
            df['r_as'] = (df['r_px'].to_numpy() * u.pix * self.arcsec_per_pixel).value
            df['r_lambda_over_d'] = characterization.arcsec_to_lambda_over_d(
                df['r_as'].to_numpy() * u.arcsec,
                self.wavelength_um * u.um,
                d=self.primary_diameter_m * u.m
            )

        log.info(f"Sampled {len(limits_df)} locations for contrast limits and detection")

        lim_xs, lim_ys = characterization.r_pa_to_x_y(limits_df[self.columns.r_px], limits_df[self.columns.pa_deg], xc, yc)
        min_r_px = np.min(limits_df[self.columns.r_px])
        iwa_pa_degs = np.unique(limits_df[limits_df[self.columns.r_px] == min_r_px][self.columns.pa_deg])
        dr_px = np.diff(np.sort(np.unique(limits_df[self.columns.r_px])))[0] / 2
        iwa_r_pxs = np.repeat(min_r_px - dr_px, len(iwa_pa_degs))
        iwa_contrasts = np.repeat(1.0, len(iwa_pa_degs))
        iwa_xs, iwa_ys = characterization.r_pa_to_x_y(iwa_r_pxs, iwa_pa_degs, xc, yc)
        lim_xs = np.concatenate([lim_xs, iwa_xs])
        lim_ys = np.concatenate([lim_ys, iwa_ys])
        lim_contrasts = np.concatenate([limits_df['contrast_limit_5sigma'], iwa_contrasts])

        contrast_lim_map = characterization.points_to_map(lim_xs, lim_ys, lim_contrasts, coverage_mask)

        det_xs, det_ys = characterization.r_pa_to_x_y(detections_df[self.columns.r_px], detections_df[self.columns.pa_deg], xc, yc)
        det_xs = np.concatenate([det_xs, iwa_xs])
        det_ys = np.concatenate([det_ys, iwa_ys])
        det_snrs = np.concatenate([detections_df['snr'], np.repeat(0, len(iwa_pa_degs))])
        detection_map = characterization.points_to_map(det_xs, det_ys, det_snrs, coverage_mask)

        hdus = [iofits.DaskHDU(None, kind="primary")]
        hdus.append(iofits.DaskHDU(contrast_lim_map, name="limits_5sigma_contrast_map"))
        hdus.append(iofits.DaskHDU(utils.convert_obj_cols_to_str(limits_df.to_records(index=False)), kind="bintable", name="limits"))
        hdus.append(iofits.DaskHDU(detection_map, name="detection_snr_map"))
        hdus.append(iofits.DaskHDU(utils.convert_obj_cols_to_str(detections_df.to_records(index=False)), kind="bintable", name="detection"))
        iofits.write_fits(iofits.DaskHDUList(hdus), output_filepath)
