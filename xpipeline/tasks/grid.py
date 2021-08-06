from .. import utils
from dataclasses import dataclass
import typing

@dataclass
class ParameterValue:
    name : str
    value : typing.Any
    formatted_arg : str

DEFAULT_FORMAT = "{name}={value}"
END_FORMAT = "destination=./{parameter_hash}/"

class Parameter:
    def __init__(self, name, values, preceding_param=None, format=DEFAULT_FORMAT, test=None):
        self.name = name
        self.values = values
        self.format = format
        self.preceding_param = preceding_param if preceding_param is not None else Grid()
        if test is not None:
            self.test = test
    def chain(self, instance):
        instance.preceding_param = self
        return instance
    def end(self, format=END_FORMAT):
        return End(self, format=format)
    def test(self, parts):
        return True
    def choices(self):
        for parts in self.preceding_param.choices():
            should_include = self.test(parts)
            if should_include:
                for value in self.values:
                    new_parts = parts.copy()
                    formatted_arg = self.format.format(name=self.name, value=value, parts=new_parts)
                    this_param_value = ParameterValue(name=self.name, value=value, formatted_arg=formatted_arg)
                    new_parts[self.name] = this_param_value
                    yield new_parts
            else:
                new_parts = parts.copy()
                this_param_value = ParameterValue(name=self.name, value=None, formatted_arg="")
                new_parts[self.name] = this_param_value
                yield new_parts
    def count(self):
        return len(list(self.choices()))
    def choices_as_args(self, prefix=""):
        for entry in self.choices():
            if len(prefix):
                parts = [prefix]
            else:
                parts = []
            for _, value in entry.items():
                if len(value.formatted_arg):
                    parts.append(value.formatted_arg)
            yield ' '.join(parts)
    def choices_as_csv(self):
        field_order = None
        for entry in self.choices():
            if field_order is None:
                field_order = list(entry.keys())
                yield ','.join(field_order)
            parts = []
            for key in field_order:
                value = entry[key].value
                parts.append(str(value) if value is not None else "")
            yield ','.join(parts)

class SilentParameter(Parameter):
    def __init__(self, name, values, preceding_param=None, test=None):
        super().__init__(name, values, preceding_param=preceding_param, format="", test=test)

class MultiParameter(Parameter):
    """Accepts sequences for name and value, yielding the i'th element of each value sequence in value"""
    def choices(self):
        for parts in self.preceding_param.choices():
            if self.test(parts):                
                for value_tuple in zip(*self.values):
                    new_parts = parts.copy()
                    for idx, a_value in enumerate(value_tuple):
                        a_name = self.name[idx]
                        formatted_arg = self.format.format(name=a_name, value=a_value, parts=new_parts)
                        new_parts[a_name] = ParameterValue(name=a_name, value=a_value, formatted_arg=formatted_arg)
                    yield new_parts
            else:
                yield parts

class Grid(Parameter):
    def __init__(self):
        super().__init__(None, None, self)
    def choices(self):
        yield {}

class End(Parameter):
    def __init__(self, preceding_param=None, format=END_FORMAT):
        super().__init__(None, None, preceding_param=preceding_param, format=format)
    def choices(self):
        for parts in self.preceding_param.choices():
            serialized = str([param.formatted_arg for param in parts.values()])
            parameter_hash = utils.str_to_sha1sum(' '.join(serialized))
            new_parts = parts.copy()
            new_parts['parameter_hash'] = ParameterValue('parameter_hash', value=parameter_hash, formatted_arg=self.format.format(parameter_hash=parameter_hash))
            yield new_parts

def parameter_choices_as_args(param, prefix=""):
    for entry in param.choices():
        if len(prefix):
            parts = [prefix]
        else:
            parts = []
        for _, value in entry.items():
            if len(value.formatted_arg):
                parts.append(value.formatted_arg)
        yield ' '.join(parts)

def parameter_choices_as_csv(param):
    field_order = None
    for entry in param.choices():
        if field_order is None:
            field_order = list(entry.keys())
            yield ','.join(field_order)
        parts = []
        for key in field_order:
            parts.append(str(entry[key].value))
        yield ','.join(parts)
