import orjson
from ..utils import str_to_sha1sum
from dataclasses import dataclass

__all__ = (
    'Category', 'Rule', 'Makeflow', 'flow_to_dict', 'save_flow'
)

@dataclass
class Category:
    cores : int
    memory_mb : int
    disk_mb : int

@dataclass
class Rule:
    command_parts : str
    inputs : list[str]
    output_basenames : list[str]
    category : str = "default"

    @property
    def destination(self):
        important_parts = " ".join(tuple(self.command_parts) + tuple(self.inputs))
        return f"outputs/{str_to_sha1sum(important_parts)}"



    @property
    def command(self):
        return " ".join(tuple(self.command_parts) + (f"destination={self.destination}/",))

    def __getitem__(self, key):
        return self.outputs[key]

    def __post_init__(self):
        if any(["destination=" in part for part in self.command_parts]):
            raise ValueError("Destination already set")
        
    @property
    def outputs(self):
        return {out: f"{self.destination}/{out}" for out in self.output_basenames}

@dataclass
class Makeflow:
    rules: list[Rule]
    categories: dict[Category] = None
    def __post_init__(self):
        if self.categories is None:
            self.categories = {'default': Category(cores=1, memory_mb=512, disk_mb=1024)}

def flow_to_dict(flow):
    payload = {}
    payload["categories"] = {k: {'cores': v.cores, 'memory': v.memory_mb, 'disk': v.disk_mb} for k, v in flow.categories.items()}
    payload["rules"] = []
    for rule in flow.rules:
        rule_payload = {}
        rule_payload['command'] = rule.command
        rule_payload['inputs'] = rule.inputs
        rule_payload['outputs'] = [v for _, v in rule.outputs.items()]
        payload["rules"].append(rule_payload)
    return payload


def save_flow(flow, dest_file):
    payload_bytes = orjson.dumps(
        flow_to_dict(flow),
        option=orjson.OPT_APPEND_NEWLINE | orjson.OPT_INDENT_2
    )
    with open(dest_file, 'wb') as f:
        f.write(payload_bytes)
        f.write(b'\n')
