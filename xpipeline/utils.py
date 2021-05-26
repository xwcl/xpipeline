import re
from urllib.parse import urlparse
import fsspec

from exao_dap_client.data_store import get_fs, join, basename

WHITESPACE_RE = re.compile(r"\s+")


def unwrap(message):
    return WHITESPACE_RE.sub(" ", message).strip()


def get_memory_use_mb():
    import os, psutil

    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def parse_obs_method(obs_method_str):
    obsmethod = {}
    for x in obs_method_str.split():
        lhs, rhs = x.split("=")
        dest = obsmethod
        parts = lhs.split(".")
        for part in parts[:-1]:
            if part not in dest:
                dest[part] = {}
            dest = dest[part]
        dest[parts[-1].lower()] = rhs
    return obsmethod


def _flatten_obs_method(the_dict):
    out = []
    for key, value in the_dict.items():
        if isinstance(value, dict):
            out.extend([f"{key}.{x}" for x in _flatten_obs_method(value)])
        else:
            out.append(f"{key.lower()}={value}")
    return out


def flatten_obs_method(obs_method):
    return " ".join(_flatten_obs_method(obs_method))
