import re
import os
import os.path
from urllib.parse import urlparse
import fsspec
import threading

_LOCAL = threading.local()
_LOCAL.filesystems = {}
WHITESPACE_RE = re.compile(r"\s+")

def join(*args):
    res = urlparse(args[0])
    new_path = os.path.join(res.path, *args[1:])
    return args[0].replace(res.path, new_path)

def basename(path):
    res = urlparse(path)
    return os.path.basename(res.path)

def get_fs(path) -> fsspec.spec.AbstractFileSystem:
    '''Obtain a concrete fsspec filesystem from a path using
    the protocol string (if any; defaults to 'file:///') and
    `_get_kwargs_from_urls` on the associated fsspec class. The same
    instance will be returned if the same kwargs are used multiple times
    in the same thread.

    Note: Won't work when kwargs are required but not encoded in the
    URL.
    '''
    scheme = urlparse(path).scheme
    proto = scheme if scheme != '' else 'file'
    cls = fsspec.get_filesystem_class(proto)
    if hasattr(cls, '_get_kwargs_from_urls'):
        kwargs = cls._get_kwargs_from_urls(path)
    else:
        kwargs = {}
    key = (proto,) + tuple(kwargs.items())
    if not hasattr(_LOCAL, 'filesystems'):
        _LOCAL.filesystems = {}   # unclear why this is not init at import in dask workers
    if key not in _LOCAL.filesystems:
        fs = cls(**kwargs)
        _LOCAL.filesystems[key] = fs
    return _LOCAL.filesystems[key]


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

def available_cpus() -> int:
    if 'OMP_NUM_THREADS' in os.environ:
        log.debug(f'Counting CPUs from OMP_NUM_THREADS')
        cpus = int(os.environ['OMP_NUM_THREADS'])
    else:
        log.debug(f'Counting CPUs from os.cpu_count')
        cpus = os.cpu_count()
    log.debug(f'Found {cpus=}')
    return cpus
