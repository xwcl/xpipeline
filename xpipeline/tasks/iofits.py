"""Loading and saving FITS with delayed functions and simplified data structures

Astropy FITS support creates objects with memory-mapped arrays and locks,
neither of which can be shipped over the network for distributed execution.
This module reads `astropy.io.fits.HDUList` objects into `DaskHDUList`
objects, converting `numpy.ndarray` data arrays to `dask.array` data arrays
(but reusing `astropy.io.fits.header.Header`)
"""
import os
import getpass
import fsspec
from fsspec.implementations.local import LocalFileSystem
from .. import utils
from distributed.protocol import register_generic
import numpy as np
from astropy.io import fits
from collections import defaultdict
import dask.array as da
import dask
import logging
import warnings

log = logging.getLogger(__name__)

_IGNORED_METADATA_KEYWORDS = set(
    (
        "BITPIX",
        "COMMENT",
        "HISTORY",
        "SIMPLE",
        "XTENSION",
        "EXTEND",
        "GCOUNT",
        "PCOUNT",
        "EXTNAME",
    )
)


def _is_ignored_metadata_keyword(kw):
    return kw in _IGNORED_METADATA_KEYWORDS or kw.startswith("NAXIS")


def _construct_dtype(varying_kw, columns):
    dtype = []
    for kw in varying_kw:
        example = columns[kw][0]
        kind = type(example)
        if kind is str:
            max_length = max([len(entry) for entry in columns[kw]])
            dtype.append((kw, kind, max_length))
        else:
            dtype.append((kw, kind))
    return dtype


def separate_varying_header_keywords(all_headers):
    columns = defaultdict(list)

    for header in all_headers:
        for kw in header.keys():
            if not _is_ignored_metadata_keyword(kw):
                columns[kw].append(header[kw])

    non_varying_kw = set()
    varying_kw = set()
    for kw, val in columns.items():
        n_different = np.unique(val).size
        if n_different == 1 and not _is_ignored_metadata_keyword(kw):
            non_varying_kw.add(kw)
        else:
            if len(val) != len(all_headers):
                log.warn(f"mismatched lengths {kw=}: {len(val)=}, {len(all_headers)}")
            varying_kw.add(kw)
    varying_dtypes = _construct_dtype(varying_kw, columns)
    return non_varying_kw, varying_kw, varying_dtypes


def construct_headers_table(all_headers):
    """Given an iterable of astropy.io.fits.Header objects, identify
    repeated and varying keywords, extracting the former to a
    "static header" and the latter to a structured array

    Returns
    -------
    static_header : astropy.io.fits.Header
    tbl_data : np.recarray
    """
    non_varying_kw, varying_kw, varying_dtypes = separate_varying_header_keywords(
        all_headers
    )
    # for keywords that vary:
    tbl_data = np.zeros(len(all_headers), varying_dtypes)
    for idx, header in enumerate(all_headers):
        for kw in varying_kw:
            if kw in header:
                tbl_data[idx][kw] = header[kw]

    # for keywords that are static:
    static_header = fits.Header()
    for card in all_headers[0].cards:
        if card.keyword in non_varying_kw:
            static_header.append(card)

    return static_header, tbl_data


class DaskHDU:
    """Represent a FITS header-data unit in a way that is
    convenient for Dask to serialize and deserialize
    """

    def __init__(self, data, header=None, kind="image"):
        if header is None:
            header = {}
        self.data = data
        self.header = fits.Header(header)
        self.kind = kind

    def __repr__(self):
        shape_dtype = (
            f"{self.data.shape} {self.data.dtype} " if self.data is not None else ""
        )
        return f"<{self.__class__.__name__}: {shape_dtype}{self.kind=}>"

    @classmethod
    def from_fits(cls, hdu, distributed=False):
        data = np.asarray(hdu.data).byteswap().newbyteorder()
        if distributed:
            data = da.from_array(data)
        header = hdu.header
        if isinstance(hdu, fits.ImageHDU):
            kind = "image"
        elif isinstance(hdu, fits.BinTableHDU):
            kind = "bintable"
        elif isinstance(hdu, fits.PrimaryHDU):
            kind = "primary"
        else:
            raise ValueError(f"Cannot handle instance of {type(hdu)}")
        this_hdu = cls(data, header, kind=kind)
        return this_hdu

    def copy(self):
        return self.__class__(self.data.copy(), self.header.copy())

    def updated_copy(self, new_data, new_headers=None, history=None):
        new_header = self.header.copy()
        if new_headers is not None:
            new_header.update(new_headers)
        if history is not None:
            new_header.add_history(history)
        return self.__class__(new_data, new_header)

    def to_fits(self):
        if self.kind == "image":
            return fits.ImageHDU(self.data, self.header)
        elif self.kind == "bintable":
            return fits.BinTableHDU(self.data, self.header)
        elif self.kind == "primary":
            return fits.PrimaryHDU(self.data, self.header)
        else:
            raise ValueError(f"Unknown kind: {self.kind}")


register_generic(DaskHDU)


def _check_ext(key, hdu, idx):
    if "EXTNAME" in hdu.header and hdu.header["EXTNAME"] == key:
        return True
    if idx == key:
        return True
    return False


class DaskHDUList:
    """Represent a list of FITS header-data units in a way that is
    convenient for Dask to serialize and deserialize
    """

    def __init__(self, hdus=None):
        if hdus is None:
            hdus = []
        self.hdus = hdus

    def append(self, hdu):
        self.hdus.append(hdu)

    def extend(self, hdus):
        self.hdus.extend(hdus)

    @classmethod
    def from_fits(cls, hdus, distributed=False):
        this_hdul = cls()
        for hdu in hdus:
            this_hdul.hdus.append(DaskHDU.from_fits(hdu, distributed=distributed))
        return this_hdul

    def to_fits(self):
        hdul = fits.HDUList([])
        for imghdu in self.hdus:
            hdul.append(imghdu.to_fits())
        return hdul

    def copy(self):
        return self.__class__([hdu.copy() for hdu in self.hdus])

    def updated_copy(self, new_data, new_headers=None, history=None, ext=0):
        """Return a copy of the DaskHDUList with extension `ext`'s data
        replaced by `new_data`, optionally appending a history card
        with what update was performed
        """
        new_hdus = []
        for idx, hdu in enumerate(self.hdus):
            if _check_ext(ext, hdu, idx):
                new_hdu = hdu.updated_copy(
                    new_data, new_headers=new_headers, history=history
                )
                new_hdus.append(new_hdu)
            else:
                new_hdus.append(hdu)
        return self.__class__(new_hdus)

    def __contains__(self, key):
        if isinstance(key, int) and key < len(self.hdus):
            return True
        else:
            for hdu in self.hdus:
                if "EXTNAME" in hdu.header and hdu.header["EXTNAME"] == key:
                    return True
        return False

    def __getitem__(self, key):
        for idx, hdu in enumerate(self.hdus):
            if _check_ext(key, hdu, idx):
                return hdu
        raise KeyError(f"No HDU {key}")


register_generic(DaskHDUList)


def load_fits(file_handle):
    hdul = fits.open(file_handle, mode="readonly", memmap=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log.debug(f"Validating FITS headers")
        hdul.verify("fix")
        for hdu in hdul:
            hdu.header.add_history("xpipeline loaded and validated format")
    log.debug(f"Converting to DaskHDUList")
    dask_hdul = DaskHDUList.from_fits(hdul)
    log.debug(f"Loaded {file_handle}: {dask_hdul.hdus}")
    return dask_hdul


def load_fits_from_path(url_or_path):
    log.debug(f"Loading {url_or_path}")
    fs = utils.get_fs(url_or_path)
    if isinstance(fs, LocalFileSystem):
        # Workaround for https://github.com/astropy/astropy/issues/11586
        path = fs._strip_protocol(url_or_path)
        with open(path, "rb") as file_handle:
            return load_fits(file_handle)
    else:
        with fsspec.open(url_or_path, "rb") as file_handle:
            return load_fits(file_handle)


def write_fits(hdul, destination_path, overwrite=False):
    log.debug(f"Writing to {destination_path}")
    fs = utils.get_fs(destination_path)
    exists = fs.exists(destination_path)
    if not overwrite and exists:
        raise RuntimeError(f"Found existing file at {destination_path}")

    with fs.open(destination_path, mode="wb") as destfh:
        log.info(f"Writing FITS HDUList to {destination_path}")
        real_hdul = hdul.to_fits()
        # OSG jobs have a variable OSGVO_SUBMITTER=josephlong@services.ci-connect.net
        if "OSGVO_SUBMTITER" in os.environ:
            pipeline_user = os.environ["OSGVO_SUBMITTER"].split("@")[0]
        else:
            pipeline_user = getpass.getuser()
        history_msg = f"written from xpipeline by {pipeline_user}"
        log.debug(history_msg)
        real_hdul[0].header.add_history(history_msg)
        real_hdul.writeto(destfh)
    return destination_path


def ensure_dq(hdul, like_ext=0):
    if "DQ" in hdul:
        log.debug("Existing DQ extension found")
        return hdul
    dq_data = np.zeros_like(hdul[like_ext].data, dtype=np.uint8)
    dq_header = {"EXTNAME": "DQ"}
    hdul.hdus.append(DaskHDU(dq_data, header=dq_header))
    msg = f"Created DQ extension based on extension {like_ext}"
    log.info(msg)
    hdul["DQ"].header.add_history(msg)
    return hdul


@dask.delayed
def _pick_ext_keyword(all_hduls, ext, keyword):
    return [hdul[ext].header[keyword] for hdul in all_hduls]


def hdulists_to_dask_cube(all_hduls, plane_shape, ext=0, dtype=float):
    cube = da.stack(
        [
            da.from_delayed(hdul[ext].data, shape=plane_shape, dtype=dtype)
            for hdul in all_hduls
        ]
    )
    log.info(f"Dask Array of shape {cube.shape} created from HDULists")
    return cube


@dask.delayed
def _kw_to_0d_seq(hdul, ext, keyword):
    return np.asarray(hdul[ext].header[keyword])


def hdulists_keyword_to_dask_array(all_hduls, keyword, ext=0, dtype=float):
    arr = da.stack(
        [
            da.from_delayed(_kw_to_0d_seq(hdul, ext, keyword), shape=(), dtype=dtype)
            for hdul in all_hduls
        ]
    )
    log.info(f"Header keyword {keyword} extracted to new {arr.shape} sequence")
    return arr
