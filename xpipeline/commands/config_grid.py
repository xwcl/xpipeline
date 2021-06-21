import enum
from humanfriendly.terminal import output
import numpy as np
import logging
from typing import Optional, Union, Any
import xconf
from itertools import product

log = logging.getLogger(__name__)

from enum import Enum

class ParamType(Enum):
    INT = 'int'
    FLOAT = 'float'

@xconf.config
class ListParam:
    values : list[Any] = xconf.field(help="List of values of the appropriate type for this option")

@xconf.config
class GridParam:
    start : Union[float,int] = xconf.field(help="Value of first point")
    stop : Union[float,int] = xconf.field(help="Value of last point")
    n_points : int = xconf.field(help="Number of values")
    type : ParamType = xconf.field(default=ParamType.FLOAT, help="Type hint for generated sequence")
    log_space : bool = xconf.field(default=False, help="Whether points should be logarithmically spaced")

    @property
    def values(self):
        if self.type is ParamType.INT:
            dtype = int
        else:
            dtype = float
        return np.linspace(self.start, self.stop, num=self.n_points, dtype=dtype)

PARAM_TYPES = [ListParam, GridParam]

def param_specs_from_dict(the_dict, path=()):
    # every key either has a gridparam, listparam, or a hash table with keys that do
    specs = {}
    for key, value in the_dict.items():
        new_path = path + (key,)
        reconstituted_value = None
        for cls in PARAM_TYPES:
            try:
                reconstituted_value = xconf.from_dict(cls, value)
            except (xconf.UnexpectedDataError, xconf.MissingValueError) as e:
                pass
        if reconstituted_value is not None:
            specs[new_path] = reconstituted_value
        elif hasattr(value, 'items'):
            specs.update(param_specs_from_dict(value, path=new_path))
        else:
            raise ValueError(f"Could not understand {value=} at {'.'.join(new_path)}")
    return specs

@xconf.config
class ConfigGrid(xconf.Command):
    """Generate command lines for a grid of parameters"""
    grid_config : str = xconf.field(help="Path to grid configuration file")
    existing_config : Optional[str] = xconf.field(help="Path to existing config to be passed with '-c' before grid parameters")

    def main(self):
        import toml
        
        with open(self.grid_config) as fh:
            grid_specs_dict = toml.load(fh)
        param_specs = param_specs_from_dict(grid_specs_dict)

        keys = []
        values = []
        for k, v in param_specs.items():
            keys.append('.'.join(k))
            values.append(v.values)
        
        output_lines = []

        for prod in product(*values):
            line_parts = []
            if self.existing_config is not None:
                line_parts.append(f'-c {self.existing_config}')
            for idx, val in enumerate(prod):
                line_parts.append(f'{keys[idx]}={val}')
            output_lines.append(' '.join(line_parts))
        for line in output_lines:
            print(line)

        # output_filepath = utils.join(destination, utils.basename("scale_templates.txt"))
        # self.quit_if_outputs_exist([output_filepath])

        # with dest_fs.open(output_filepath, 'w') as fh:
        #     for factor in factors:
        #         fh.write(f'{factor}\n')
