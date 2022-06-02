import numpy as np
import xconf
from xconf.contrib import BaseRayGrid, FileConfig, join, PathConfig, DirectoryConfig
from ..pipelines.new import MeasureStarlightSubtraction, StarlightSubtractionData
from ..constants import KlipStrategy

def _measure_subtraction_task(
    measure_subtraction: MeasureStarlightSubtraction,
    data: StarlightSubtractionData
):
    return measure_subtraction.execute(data)


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
    sampling : SamplingConfig = xconf.field(help="")
    included_annuli_resel : list[float] = xconf.field(
        default_factory=lambda: [0, 2, 4],
        help="examine the effect of a more-restrictive annular mask of X lambda/D about the location of interest, 0 = no additional mask"
    )

    def compare_grid_to_checkpoint(self, checkpoint_tbl: np.ndarray, grid_tbl: np.ndarray) -> bool:
        raise NotImplementedError(
            "Subclasses must implement compare_grid_to_checkpoint()"
        )

    def generate_grid(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement generate_grid()")

    def launch_grid(self, pending_tbl: np.ndarray) -> list:
        """Launch Ray tasks for each grid point and collect object
        refs. The Ray remote function ref must return a copy of the
        grid row it's called with, updating 'time_total_sec' to
        indicate it's been processed.
        """
        import ray
        measure_subtraction_task = ray.remote(_measure_subtraction_task)
        raise NotImplementedError("Subclasses must implement launch_grid()")
