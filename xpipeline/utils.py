import re

WHITESPACE_RE = re.compile(r"\s+")


def unwrap(message):
    return WHITESPACE_RE.sub(" ", message)
