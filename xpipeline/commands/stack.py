import logging
from collections import defaultdict
from typing import Optional, Union
import xconf
from .. import utils, constants, types
from .base import BaseCommand
from .base import CompanionConfig, TemplateConfig, AngleRangeConfig, PixelRotationRangeConfig, FitsConfig, FitsTableColumnConfig
from dataclasses import dataclass

log = logging.getLogger(__name__)

import numpy as np

@xconf.config
class ComponentConfig(FitsConfig):
    derotate_by : Optional[FitsTableColumnConfig] = xconf.field(default=None, help="Metadata table column with derotation angle in degrees")
    mask : Optional[FitsConfig] = xconf.field(default=None)
    destination_idx : int = xconf.field(default=0)
    amplitude_scale : float = xconf.field(default=1, help="Factor by which pixels in this component are multiplied before adding to the stack")

@xconf.config
class Stack(BaseCommand):
    stack_by : constants.CombineOperation = xconf.field(default=constants.CombineOperation.SUM, help="Operation used to stack final derotated data")
    components : list[ComponentConfig] = xconf.field()

    def main(self):
        import fsspec.spec
        from ..tasks import iofits
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {destination}")
        dest_fs.makedirs(destination, exist_ok=True)

        output_filepath = utils.join(destination, utils.basename("stack.fits"))
        if dest_fs.exists(output_filepath):
            raise FileExistsError(f"{output_filepath} exists")

        outputs_to_components = {}
        for comp in self.components:
            if comp.destination_idx not in outputs_to_components:
                outputs_to_components[comp.destination_idx] = [comp]
            else:
                outputs_to_components[comp.destination_idx].append(comp)
        
        hdus = []
        for output_idx, components in outputs_to_components.items():
            output_stacked, coverage_stacked = self._process_one_output(components)
            hdus.append(iofits.DaskHDU(output_stacked, name=f"STACK_{output_idx}"))
            hdus.append(iofits.DaskHDU(coverage_stacked, name=f"COVERAGE_{output_idx}"))
        
        iofits.write_fits(iofits.DaskHDUList(hdus), output_filepath)

            
    def _process_one_output(self, components : list[ComponentConfig]):
        from ..tasks import improc
        output_data = None
        output_coverage = None
        for comp_idx, comp in enumerate(components):
            comp_data = comp.load()
            n1, nrest = comp_data.shape[0], comp_data.shape[1:]
            if output_data is None:
                overall_shape = (n1 * len(components),) + nrest
                output_data = np.zeros(overall_shape, dtype=comp_data.dtype)
                output_coverage = np.zeros(overall_shape)
            if comp.mask is not None:
                combo_mask = comp.mask.load().astype(bool)
            else:
                combo_mask = np.ones_like(comp_data[0], dtype=bool)
            masked_data = np.nan * np.ones_like(comp_data)
            masked_data[:,combo_mask] = comp_data[:,combo_mask]
            coverage_cube = np.repeat(combo_mask[np.newaxis, :, :], masked_data.shape[0], axis=0)
            if comp.derotate_by is not None:
                derotation_angles = comp.derotate_by.load()
                masked_data = improc.derotate_cube(masked_data, derotation_angles)
                coverage_cube = improc.derotate_cube(coverage_cube, derotation_angles)
            output_data[comp_idx*n1:(comp_idx+1)*n1] = masked_data
            output_coverage[comp_idx*n1:(comp_idx+1)*n1] = coverage_cube
        return improc.combine(output_data, operation=self.stack_by), improc.combine(output_coverage, constants.CombineOperation.SUM)
