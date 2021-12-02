import re
import hashlib
import numpy as np
import os
import os.path
from urllib.parse import urlparse
import fsspec
import threading
import logging
import numba
import psutil

from .core import HAVE_CUPY

log = logging.getLogger(__name__)

_LOCAL = threading.local()
_LOCAL.filesystems = {}
WHITESPACE_RE = re.compile(r"\s+")

class DummyRamHook:
    used_bytes = 0
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass

if HAVE_CUPY:
    from cupy.cuda import memory_hook, runtime
    import cupy

    class CupyRamHook(memory_hook.MemoryHook):
        def __init__(self):
            self.used_bytes = cupy.get_default_memory_pool().used_bytes()
        def alloc_preprocess(self, device_id, mem_size):
            self.used_bytes += mem_size
else:
    CupyRamHook = DummyRamHook

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


@numba.njit([(the_float[:,:], numba.intp, numba.intp) for the_float in (numba.float32, numba.float64)], cache=True)
def drop_idx_range_rows_cols(arr, min_excluded_idx, max_excluded_idx):
    '''Note exclusive upper bound: [min_excluded_idx, max_excluded_idx)'''
    rows, cols = arr.shape
    assert rows == cols
    n_drop = max_excluded_idx - min_excluded_idx
    out_shape = (rows - n_drop, cols - n_drop)
    out = np.empty(out_shape, dtype=arr.dtype)
    # UL | | U R
    # ===   ====
    # LL | | L R

    # UL
    out[:min_excluded_idx,:min_excluded_idx] = arr[:min_excluded_idx,:min_excluded_idx]
    # UR
    out[:min_excluded_idx,min_excluded_idx:] = arr[:min_excluded_idx,max_excluded_idx:]
    # LL
    out[min_excluded_idx:,:min_excluded_idx] = arr[max_excluded_idx:,:min_excluded_idx]
    # LR
    out[min_excluded_idx:,min_excluded_idx:] = arr[max_excluded_idx:,max_excluded_idx:]
    return out

@numba.njit([(the_float[:,:], numba.intp, numba.intp) for the_float in (numba.float32, numba.float64)], cache=True)
def drop_idx_range_rows(arr, min_excluded_idx, max_excluded_idx):
    '''Note exclusive upper bound: [min_excluded_idx, max_excluded_idx)'''
    rows, cols = arr.shape
    if min_excluded_idx < 0:
        raise ValueError("Negative indexing unsupported")
    if max_excluded_idx > rows or max_excluded_idx < min_excluded_idx:
        raise ValueError("Upper limit of excluded indices out of bounds")
    n_drop = max_excluded_idx - min_excluded_idx
    out_shape = (rows - n_drop, cols)
    out = np.empty(out_shape, dtype=arr.dtype)
    #  U
    # ===
    #  L

    # U
    out[:min_excluded_idx] = arr[:min_excluded_idx]
    # L
    out[min_excluded_idx:] = arr[max_excluded_idx:]
    return out

@numba.njit([(the_float[:,:], numba.intp, numba.intp) for the_float in (numba.float32, numba.float64)], cache=True)
def drop_idx_range_cols(arr, min_excluded_idx, max_excluded_idx):
    '''Note exclusive upper bound: [min_excluded_idx, max_excluded_idx)'''
    return drop_idx_range_rows(arr.T, min_excluded_idx, max_excluded_idx).T

def str_to_sha1sum(string):
    hasher = hashlib.sha1(string.encode('utf8'))
    return hasher.hexdigest()

def num_cpus():
    '''Return number of CPUs reported by the OS, or the
    number available based on CPU affinity, if smaller'''
    count = os.cpu_count()
    try:
        cpus_affinity = len(psutil.Process().cpu_affinity())
        if cpus_affinity > 0:
            count = min(count, cpus_affinity)
    except Exception:
        pass
    return count

CPU_COUNT = num_cpus()

from matplotlib.cm import magma
gmagma = magma.copy()
gmagma.set_bad('gray')
from pkg_resources import packaging
def version_greater_or_equal(version_a, version_b):
    return packaging.version.parse(version_a) >= packaging.version.parse(version_b)
