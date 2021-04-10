import re
from urllib.parse import urlparse
import fsspec

from exao_dap_client.data_store import get_fs, join, basename

WHITESPACE_RE = re.compile(r"\s+")


def unwrap(message):
    return WHITESPACE_RE.sub(" ", message)
