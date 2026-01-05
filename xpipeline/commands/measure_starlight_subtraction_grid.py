import time
import numpy as np
import xconf
import logging
from typing import Optional
from xconf.contrib import BaseRayGrid, DirectoryConfig
import ray
from ray._raylet import ObjectRef
from ..commands.base import AnyRayConfig, LocalRayConfig, measure_ram
from ..pipelines.new import (
    MeasureStarlightSubtraction, StarlightSubtractionData, KModesConfig,
    KModesFractionConfig, KModesValuesConfig, StarlightSubtractionDataConfig,
    StarlightSubtractionMeasurement, StarlightSubtractionFilterMeasurements,
    SaveMeasuredStarlightSubtraction, CompanionConfig, StarlightSubtractionFilterMeasurement
)
from ..constants import KlipStrategy
from ..tasks import characterization

log = logging.getLogger(__name__)

def _measure_subtraction_task(
    chunk: np.ndarray,
    measure_subtraction: MeasureStarlightSubtraction,
    save_subtraction: SaveMeasuredStarlightSubtraction,
    data_config: StarlightSubtractionDataConfig,
    destination: DirectoryConfig,
    save_intermediates: bool = False
):
    # apply configuration for chunk points
    k_modes_requested = list(sorted(np.unique(chunk['k_modes_requested'])))
    if np.issubdtype(chunk['k_modes_requested'].dtype, float):
        k_modes_spec = KModesFractionConfig(fractions=k_modes_requested)
    else:
        k_modes_spec = KModesValuesConfig(values=k_modes_requested)
    measure_subtraction.subtraction.k_modes = k_modes_spec
    data_config.companions = [CompanionConfig(
        r_px=chunk[0]['r_px'],
        pa_deg=chunk[0]['pa_deg'],
        scale=chunk[0]['injected_scale'],
    )]

    data_config.decimate_frames_by = chunk[0]['decimate_frames_by']
    resel_px = measure_subtraction.subtraction.resolution_element_px
    annulus_resel = chunk[0]['annulus_resel']
    if annulus_resel > 0:
        for idx in range(len(data_config.inputs)):
            data_config.inputs[idx].radial_mask.min_r_px = data_config.companions[0].r_px - (annulus_resel * resel_px) / 2
            data_config.inputs[idx].radial_mask.max_r_px = data_config.companions[0].r_px + (annulus_resel * resel_px) / 2
    log.debug(f'''Configured:
{measure_subtraction.subtraction.k_modes=}
{data_config.companions[0].r_px=}
{data_config.companions[0].pa_deg=}
{data_config.companions[0].scale=}
{data_config.inputs[0].radial_mask.min_r_px=}, {data_config.inputs[0].radial_mask.max_r_px=}
''')
    # measure starlight subtraction
    start = time.perf_counter()
    log.debug(f"Starting measure subtraction task at {start=}")
    data = data_config.load()

    if save_intermediates:
        intermed_path = destination.join(f"chunk_{np.min(chunk['index'])}-{np.max(chunk['index'])}")
        meas = save_subtraction.execute(data, measure_subtraction, destination=DirectoryConfig(path=intermed_path))
    else:
        meas = measure_subtraction.execute(data)
    elapsed = time.perf_counter() - start
    log.debug(f"Completed measure subtraction task in {elapsed} sec from {start=}")

    # update chunk entries with measurements
    k_modes_chosen = list(sorted(meas.by_modes.keys()))
    k_modes_requested_to_chosen = {meas.modes_chosen_to_requested_lookup[k]: k for k in k_modes_chosen}
    chunk = chunk.copy()
    chunk['time_total_sec'] = elapsed / len(chunk)   # at the end, column should add up to total execution time
    for idx in range(len(chunk)):
        ext = chunk[idx]['ext'].decode('utf8')
        filter_name = chunk[idx]['filter_name'].decode('utf8')
        k_modes_for_row = chunk[idx]['k_modes_requested']
        if k_modes_for_row not in k_modes_requested_to_chosen:
            # skipped because it was > the max modes available
            # put a sentinel value in k_modes_chosen (not NaN because it's int)
            # and NaNs in the measurements
            chunk[idx]['k_modes_chosen'] = -1
            chunk[idx]['signal'] = np.nan
            chunk[idx]['snr'] = np.nan
        else:
            k_modes_chosen_for_row = k_modes_requested_to_chosen[k_modes_for_row]
            chunk[idx]['k_modes_chosen'] = k_modes_chosen_for_row
            measurements : StarlightSubtractionFilterMeasurements = meas.by_modes[k_modes_chosen_for_row].by_ext[ext]
            filter_measurement : StarlightSubtractionFilterMeasurement = getattr(measurements, filter_name)

            chunk[idx]['signal'] = filter_measurement.locations[0].signal
            chunk[idx]['snr'] = filter_measurement.locations[0].snr
    return chunk


@xconf.config
class SamplingConfig:
    n_radii : int = xconf.field(help="Number of steps in radius at which to probe contrast")
    spacing_px : float = xconf.field(help="Spacing in pixels between contrast probes along circle (sets number of probes at radius by 2 * pi * r / spacing)")
    scales : list[float] = xconf.field(default_factory=lambda: [0.0], help="Probe contrast levels (C = companion / host)")
    iwa_px : float = xconf.field(help="Inner working angle (px)")
    owa_px : float = xconf.field(help="Outer working angle (px)")

    def __post_init__(self):
        # to make use of this for detection, we must also apply the matched
        # filter in the no-injection case for each combination of parameters
        self.scales = [float(s) for s in self.scales]
        if 0.0 not in self.scales:
            self.scales.insert(0, 0.0)
            log.info("Inserted a 0.0 scale entry in SamplingConfig")


@xconf.config
class MeasureStarlightSubtractionGrid(BaseRayGrid):
    ray : AnyRayConfig = xconf.field(
        default_factory=LocalRayConfig,
        help="Ray distributed framework configuration"
    )
    measure_subtraction : MeasureStarlightSubtraction = xconf.field(help="")
    save_subtraction : SaveMeasuredStarlightSubtraction = xconf.field(default_factory=SaveMeasuredStarlightSubtraction, help="")
    save_intermediates : bool = xconf.field(default=False, help="Whether to run or skip the saving steps configured under 'save_subtraction'")
    data : StarlightSubtractionDataConfig = xconf.field(help="Starlight subtraction data")
    decimate_frames_by_values : list[float] = xconf.field(default_factory=lambda: [1], help="Evaluate a grid at multiple decimation levels (taking every Nth frame)")
    ram_mb_for_decimation_values : Optional[list[float]] = xconf.field(default=None, help="Amount of RAM needed at each decimation level, omit to measure")
    sampling : SamplingConfig = xconf.field(help="")
    included_annuli_resel : list[float] = xconf.field(
        default_factory=lambda: [0, 2, 4],
        help="examine the effect of a more-restrictive annular mask of X lambda/D about the location of interest "
             "(X / 2 inwards and X / 2 outwards), 0 = no additional mask"
    )

    def compare_grid_to_checkpoint(self, checkpoint_tbl: np.ndarray, grid_tbl: np.ndarray) -> bool:
        parameters = ['index', 'r_px', 'pa_deg', 'x', 'y', 'injected_scale']
        try:
            for param in parameters:
                if not np.allclose(checkpoint_tbl[param], grid_tbl[param]):
                    return False
        except Exception:
            return False
        return True

    def generate_grid(self) -> np.ndarray:
        destination_exts = set()
        for pinput in self.data.inputs:
            for destination_ext in pinput.destination_exts:
                destination_exts.add(destination_ext)
        max_len_destination_ext = max(len(f) for f in destination_exts)
        filter_names = [
            'tophat',
            'matched',
        ]
        annuli_resel = self.included_annuli_resel
        max_len_filter_name = max(len(f) for f in filter_names)
        cols_dtype = [
            ('index', int),
            ('time_total_sec', float),
            ('r_px', float),
            ('pa_deg', float),
            ('x', float),
            ('y', float),
            ('injected_scale', float),
            ('decimate_frames_by', int),
            ('annulus_resel', float),
            ('k_modes_chosen', int),
            ('snr', float),
            ('signal', float),
            ('ext', f'S{max_len_destination_ext}'),
            ('filter_name', f'S{max_len_filter_name}'), # note NumPy silently truncates strings longer than this on assignment
        ]
        if hasattr(self.measure_subtraction.subtraction.k_modes, 'fractions'):
            k_modes_choices = self.measure_subtraction.subtraction.k_modes.fractions
            cols_dtype.append(('k_modes_requested', float))
        else:
            k_modes_choices = self.measure_subtraction.subtraction.k_modes.values
            cols_dtype.append(('k_modes_requested', int))

        probes = list(characterization.generate_probes(
            self.sampling.iwa_px,
            self.sampling.owa_px,
            self.sampling.n_radii,
            self.sampling.spacing_px,
            self.sampling.scales
        ))

        n_comp_rows = (
            len(self.decimate_frames_by_values)
            * len(annuli_resel)
            * len(k_modes_choices)
            * len(destination_exts)
            * len(filter_names)
            * len(probes)
        )
        log.debug(f"Evaluating {len(probes)} positions/contrast levels at {len(k_modes_choices)} k values: {k_modes_choices}")
        comp_grid = np.zeros(n_comp_rows, dtype=cols_dtype)
        flattened_idx = 0
        for decimate_value in self.decimate_frames_by_values:
            for comp in probes:
                for k_modes in k_modes_choices:
                    for annulus_resel in annuli_resel:
                        for dest_ext in destination_exts:
                            for filter_name in filter_names:
                                comp_grid[flattened_idx]['index'] = flattened_idx
                                comp_grid[flattened_idx]['r_px'] = comp.r_px
                                comp_grid[flattened_idx]['pa_deg'] = comp.pa_deg
                                comp_grid[flattened_idx]['injected_scale'] = comp.scale
                                comp_grid[flattened_idx]['decimate_frames_by'] = decimate_value
                                comp_grid[flattened_idx]['annulus_resel'] = annulus_resel
                                comp_grid[flattened_idx]['k_modes_requested'] = k_modes
                                comp_grid[flattened_idx]['ext'] = dest_ext
                                comp_grid[flattened_idx]['filter_name'] = filter_name
                                flattened_idx += 1
        return comp_grid


    def launch_grid(self, pending_tbl: np.ndarray) -> list[ObjectRef]:
        """Launch Ray tasks for each grid point and collect object
        refs. The Ray remote function ref must return a copy of the
        grid row it's called with, updating 'time_total_sec' to
        indicate it's been processed.
        """

        measure_subtraction_task = ray.remote(_measure_subtraction_task)
        pending_refs = []
        externally_varying_params = ['r_px', 'pa_deg', 'annulus_resel', 'injected_scale', 'decimate_frames_by']
        unique_params = np.unique(pending_tbl[externally_varying_params])


        decimate_level_to_ram_bytes = {}
        if self.ram_mb_for_decimation_values is None:
            for decimate_level in np.unique(pending_tbl['decimate_frames_by']):
                expensive_mask = pending_tbl['decimate_frames_by'] == decimate_level
                if np.count_nonzero(expensive_mask) == 0:
                    log.debug(f"Skipping RAM use estimate for {decimate_level} because no points are pending")
                    continue
                need_injection_mask = pending_tbl['injected_scale'] != 0
                if np.count_nonzero(expensive_mask & need_injection_mask):
                    expensive_mask &= need_injection_mask
                expensive_grid_points = pending_tbl[expensive_mask]
                no_annulus_mask = expensive_grid_points['annulus_resel'] == 0
                if np.count_nonzero(no_annulus_mask):
                    expensive_grid_points = expensive_grid_points[no_annulus_mask]
                else:
                    expensive_grid_points = expensive_grid_points[expensive_grid_points['annulus_resel'] == np.max(expensive_grid_points['annulus_resel'])]
                assert len(expensive_grid_points) > 0
                expensive_grid_point = expensive_grid_points[:1]
                pending_modes_values = np.unique(pending_tbl['k_modes_requested'])
                expensive_grid_chunk = np.repeat(expensive_grid_point, len(pending_modes_values))
                expensive_grid_chunk['index'] = -1
                expensive_grid_chunk['k_modes_requested'] = pending_modes_values

                log.debug(f"Measure RAM requirement for {decimate_level=}...")
                ram_requirement_mb, gpu_ram_requirement_mb = measure_ram(
                    _measure_subtraction_task,
                    {},
                    expensive_grid_chunk,
                    self.measure_subtraction,
                    self.save_subtraction,
                    self.data,
                    self.destination,
                    self.save_intermediates,
                )
                ram_requirement_bytes = ram_requirement_mb * 1024 * 1024
                decimate_level_to_ram_bytes[int(decimate_level)] = ram_requirement_bytes
        else:
            for idx in range(len(self.ram_mb_for_decimation_values)):
                decimate_level_to_ram_bytes[self.decimate_frames_by_values[idx]] = self.ram_mb_for_decimation_values[idx] * 1024 * 1024
        log.debug(f"RAM usage by decimation level: {decimate_level_to_ram_bytes}")

        for combination in unique_params:
            chunk_mask = np.ones_like(pending_tbl, dtype=bool)
            for field_name in externally_varying_params:
                chunk_mask &= pending_tbl[field_name] == combination[field_name]
            assert np.count_nonzero(chunk_mask) > 0
            chunk = pending_tbl[chunk_mask]
            decimate_level = int(chunk[0]['decimate_frames_by'])
            ram_requirement_bytes = decimate_level_to_ram_bytes[decimate_level]
            ref : ObjectRef = measure_subtraction_task.options(
                memory=ram_requirement_bytes
            ).remote(
                chunk,
                self.measure_subtraction,
                self.save_subtraction,
                self.data,
                self.destination,
                self.save_intermediates,
            )
            pending_refs.append(ref)
        return pending_refs
