from enum import Enum
import enum
from humanfriendly.terminal import output
import numpy as np
import logging
from typing import Optional, Union, Any
import xconf
from itertools import product

log = logging.getLogger(__name__)


class ParamType(Enum):
    INT = 'int'
    FLOAT = 'float'


@xconf.config
class AbstractGridParam:
    start : Union[float, int] = xconf.field(help="Value of first point")
    stop : Union[float, int] = xconf.field(help="Value of last point")
    type : ParamType = xconf.field(default=ParamType.FLOAT, help="Type hint for generated sequence")

    @property
    def _numpy_dtype(self):
        if self.type is ParamType.INT:
            return np.int32
        else:
            return np.float32


@xconf.config
class StepGridParam(AbstractGridParam):
    step : Union[float, int] = xconf.field(help="Spacing between successive values")

    @property
    def values(self):
        return np.arange(self.start, self.stop, step=self.step, dtype=self._numpy_dtype)


@xconf.config
class NPointsGridParam(AbstractGridParam):
    n_points : int = xconf.field(help="Number of values")
    log_space : bool = xconf.field(default=False, help="Whether points should be logarithmically spaced")

    @property
    def values(self):
        if self.type is ParamType.INT:
            dtype = int
        else:
            dtype = float
        if self.log_space:
            return np.logspace(np.log10(self.start), np.log10(self.stop), num=self.n_points, dtype=self._numpy_dtype)
        else:
            return np.linspace(self.start, self.stop, num=self.n_points, dtype=self._numpy_dtype)


@xconf.config
class ListParam:
    values : list[Any] = xconf.field(help="List of values of the appropriate type for this option")


PARAM_TYPES = [ListParam, StepGridParam, NPointsGridParam]


def param_specs_from_dict(parse_result, path=()):
    # every key either has a gridparam, listparam, or a hash table with keys that do
    specs = {}
    print(f"{path=} {parse_result=}")
    reconstituted_value = None
    for cls in PARAM_TYPES:
        print(f"trying {cls=}")
        try:
            reconstituted_value = xconf.from_dict(cls, parse_result)
            print(reconstituted_value)
            break
        except (xconf.UnexpectedDataError, xconf.MissingValueError, AttributeError) as e:
            print(e)
            pass
    if reconstituted_value is not None:

        specs[path] = reconstituted_value
    elif isinstance(parse_result, list):
        path_base, key = path[:-1], path[-1]
        for idx, entry in enumerate(parse_result):
            new_path = path_base + (f"{key}[{idx}]",)
            specs.update(param_specs_from_dict(entry, path=new_path))
    elif hasattr(parse_result, 'items'):
        for key, value in parse_result.items():
            new_path = path + (key,)
            specs.update(param_specs_from_dict(value, path=new_path))
    else:
        raise ValueError(f"Could not understand {parse_result=} at {'.'.join(path)}")
    print(f'{specs=}')
    return specs


@xconf.config
class ConfigGrid(xconf.Command):
    """Generate command lines for a grid of parameters"""
    grid_file : str = xconf.field(help="Path to grid generation configuration file")
    existing_config : Optional[str] = xconf.field(
        help="Path to existing config to be passed with '-c' before grid parameters")

    def main(self):
        import toml

        with open(self.grid_file) as fh:
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
                if isinstance(val, float):
                    if val < 1:
                        val_str = "{:e}".format(val)
                    else:
                        val_str = "{:1.3}".format(val)
                else:
                    val_str = str(val)
                line_parts.append(f'{keys[idx]}={val_str}')
            output_lines.append(' '.join(line_parts))
        for line in output_lines:
            print(line)
