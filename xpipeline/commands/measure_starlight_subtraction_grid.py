import numpy as np
import xconf
import logging
from xconf.contrib import BaseRayGrid, FileConfig, join, PathConfig, DirectoryConfig
import ray
from ray._raylet import ObjectRef
from ..pipelines.new import (
    MeasureStarlightSubtraction, StarlightSubtractionData, KModesConfig,
    KModesFractionConfig, KModesValuesConfig, StarlightSubtractionDataConfig,
)
from ..constants import KlipStrategy
from ..tasks import characterization

log = logging.getLogger(__name__)

def _measure_subtraction_task(
    chunk: np.ndarray,
    measure_subtraction: MeasureStarlightSubtraction,
    data: StarlightSubtractionData
):
    if np.issubdtype(chunk['k_modes'].dtype, float):
        k_modes_spec = KModesFractionConfig(fractions=np.unique(chunk['k_modes']))
    else:
        k_modes_spec = KModesValuesConfig(values=np.unique(chunk['k_modes']))
    measure_subtraction.subtraction.k_modes = k_modes_spec
    meas = measure_subtraction.execute(data)
    result = measure_subtraction.measurements_to_jsonable(meas)
    # update chunk entries with measurements




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


@xconf.config
class MeasureStarlightSubtractionGrid(BaseRayGrid):
    measure_subtraction : MeasureStarlightSubtraction = xconf.field(help="")
    data : StarlightSubtractionDataConfig = xconf.field(help="Starlight subtraction data")
    sampling : SamplingConfig = xconf.field(help="")
    included_annuli_resel : list[float] = xconf.field(
        default_factory=lambda: [0, 2, 4],
        help="examine the effect of a more-restrictive annular mask of X lambda/D about the location of interest, 0 = no additional mask"
    )

    def compare_grid_to_checkpoint(self, checkpoint_tbl: np.ndarray, grid_tbl: np.ndarray) -> bool:
        parameters = ['index', 'r_px', 'pa_deg', 'x', 'y', 'injected_scale']
        for param in parameters:
            # print(param, 'chk', np.unique(checkpoint_tbl[param]))
            # print(param, 'grid_tbl', np.unique(grid_tbl[param]))
            if not np.allclose(checkpoint_tbl[param], grid_tbl[param]):
                return False
        return True

    def generate_grid(self) -> np.ndarray:
        destination_exts = set()
        for pinput in self.data.inputs:
            destination_exts.add(pinput.destination_ext)
        max_len_destination_ext = max(len(f) for f in destination_exts)
        filter_names = ['none', 'tophat', 'matched']
        max_len_filter_name = max(len(f) for f in filter_names)
        cols_dtype = [
            ('index', int),
            ('time_total_sec', float),
            ('r_px', float),
            ('pa_deg', float),
            ('x', float),
            ('y', float),
            ('injected_scale', float),
            ('snr', float),
            ('signal', float),
            ('ext', f'S{max_len_destination_ext}'),
            ('filter_name', f'S{max_len_filter_name}'), # note NumPy silently truncates strings longer than this on assignment
        ]
        if hasattr(self.measure_subtraction.subtraction.k_modes, 'fractions'):
            k_modes_choices = self.measure_subtraction.subtraction.k_modes.fractions
            cols_dtype.append(('k_modes', float))
        else:
            k_modes_choices = self.measure_subtraction.subtraction.k_modes.values
            cols_dtype.append(('k_modes', int))
        
        probes = list(characterization.generate_probes(
            self.sampling.iwa_px,
            self.sampling.owa_px,
            self.sampling.n_radii,
            self.sampling.spacing_px,
            self.sampling.scales
        ))
        n_comp_rows = len(k_modes_choices) * len(destination_exts) * len(filter_names) * len(probes)
        log.debug(f"Evaluating {len(probes)} positions/contrast levels at {len(self.k_modes_vals)} k values")
        comp_grid = np.zeros(n_comp_rows, dtype=cols_dtype)
        flattened_idx = 0
        for comp in probes:
            for dest_ext in destination_exts:
                for filter_name in filter_names:
                    # for every number of modes:
                    for k_modes in k_modes_choices:
                        comp_grid[flattened_idx]['index'] = flattened_idx
                        comp_grid[flattened_idx]['r_px'] = comp.r_px
                        comp_grid[flattened_idx]['pa_deg'] = comp.pa_deg
                        comp_grid[flattened_idx]['injected_scale'] = comp.scale
                        comp_grid[flattened_idx]['k_modes'] = k_modes
                        comp_grid[flattened_idx]['filter_name'] = filter_name
                        flattened_idx += 1
        return comp_grid

        # idx
        # r
        # pa
        # per kmodes per filter signal
        # per kmodes per filter snr
        # per kmodes kmodes
        raise NotImplementedError("Subclasses must implement generate_grid()")

    def launch_grid(self, pending_tbl: np.ndarray) -> list[ObjectRef]:
        """Launch Ray tasks for each grid point and collect object
        refs. The Ray remote function ref must return a copy of the
        grid row it's called with, updating 'time_total_sec' to
        indicate it's been processed.
        """
        
        measure_subtraction_task = ray.remote(_measure_subtraction_task)
        pending_refs = []
        for row in pending_tbl:
            ref : ObjectRef = measure_subtraction_task.remote(

            )
            pending_refs.append(ref)
        return pending_refs
