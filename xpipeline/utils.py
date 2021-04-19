import re
from urllib.parse import urlparse
import fsspec

from exao_dap_client.data_store import get_fs, join, basename

WHITESPACE_RE = re.compile(r"\s+")


def unwrap(message):
    return WHITESPACE_RE.sub(" ", message)

def get_memory_use_mb():
    import os, psutil;
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
