import datetime
import dateutil.parser
import numpy as np
from astropy.io import fits
from collections import defaultdict
import logging


from .iofits import _is_ignored_metadata_keyword

log = logging.getLogger(__name__)

def _datetime_str_to_posix_time(time_str):
    dt = dateutil.parser.parse(time_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp()

def _construct_dtype(varying_kw, columns):
    dtype = []
    dateful = []
    for kw in varying_kw:
        example = columns[kw][0]
        if isinstance(example, str):
            max_length = max([len(entry) for entry in columns[kw]])
            dtype.append((kw, str, max_length))
            try:
                _datetime_str_to_posix_time(example)
                dateful.append(kw)
            except ValueError:
                pass
        elif isinstance(example, float):
            dtype.append((kw, np.float32))
        else:
            dtype.append((kw, type(example)))
    return dtype, dateful


def separate_varying_header_keywords(all_headers):
    columns = defaultdict(list)

    for header in all_headers:
        for kw in header.keys():
            if not _is_ignored_metadata_keyword(kw):
                columns[kw].append(header[kw])

    non_varying_kw = set()
    varying_kw = set()
    for kw, val in columns.items():
        n_different = len(np.unique(val))
        if n_different == 1 and not _is_ignored_metadata_keyword(kw):
            non_varying_kw.add(kw)
        else:
            if len(val) != len(all_headers):
                log.warning(f"mismatched lengths {kw=}: {len(val)=}, {len(all_headers)=}")
            varying_kw.add(kw)
    varying_dtypes, dateful_kws = _construct_dtype(varying_kw, columns)
    return non_varying_kw, varying_kw, varying_dtypes, dateful_kws

def _date_timestamp_cols(varying_dtypes, dateful_kws):
    for kw in dateful_kws:
        for idx, field in enumerate(varying_dtypes):
            field_name = field[0]
            if field_name == kw:
                varying_dtypes.insert(idx + 1, (kw+"_TS", np.float64))
                break
    return varying_dtypes
    

def construct_headers_table(all_headers : list[fits.Header]):
    """Given an iterable of astropy.io.fits.Header objects, identify
    repeated and varying keywords, extracting the former to a
    "static header" and the latter to a structured array

    Parameters
    ----------
    all_headers : list[fits.Header]
        List of all the headers to use

    Returns
    -------
    static_header : astropy.io.fits.Header
    tbl_data : np.ndarray
    tbl_mask : np.ndarray
    """
    non_varying_kw, varying_kw, varying_dtypes, dateful_kws = separate_varying_header_keywords(
        all_headers
    )
    varying_dtypes = _date_timestamp_cols(varying_dtypes, dateful_kws)
    # for keywords that vary:
    tbl_data = np.zeros(len(all_headers), varying_dtypes)
    mask_dtypes = []
    for parts in varying_dtypes:
        mask_dtypes.append((parts[0], np.bool))
    tbl_mask = np.zeros(len(all_headers), mask_dtypes)
    for idx, header in enumerate(all_headers):
        for kw in varying_kw:
            if kw in header:
                if kw in dateful_kws:
                    tbl_data[idx][kw] = header[kw]
                    tbl_data[idx][kw+"_TS"] = _datetime_str_to_posix_time(header[kw])
                else:
                    tbl_data[idx][kw] = header[kw]
            else:
                tbl_mask[idx][kw] = True  # true where value is missing/imputed and should be masked in maskedarray

    tbl = np.ma.array(data=tbl_data, dtype=varying_dtypes, mask=tbl_mask)

    # for keywords that are static:
    static_header = fits.Header()
    for card in all_headers[0].cards:
        if card.keyword in non_varying_kw:
            static_header.append(card)

    return static_header, tbl
