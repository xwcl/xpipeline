'''Loading and saving FITS with delayed functions and simplified data structures

Astropy FITS support creates objects with memory-mapped arrays and locks,
neither of which can be shipped over the network for distributed execution.
This module reads `astropy.io.fits.HDUList` objects into `DaskHDUList` 
objects, converting `numpy.ndarray` data arrays to `dask.array` data arrays
(but reusing `astropy.io.fits.header.Header`)
'''
from copy import deepcopy
import warnings
import logging
log = logging.getLogger(__name__)

import dask
import dask.array as da
from astropy.io import fits
import numpy as np
from distributed.protocol import dask_serialize, dask_deserialize, register_generic



class DaskHDU:
    '''Represent a FITS header-data unit in a way that is
    convenient for Dask to serialize and deserialize
    '''
    def __init__(self, data, header=None):
        if header is None:
            header = {}
        self.data = np.asarray(data).copy()
        self.header = fits.Header(header)
    @classmethod
    def from_fits(cls, hdu):
        data = da.asarray(hdu.data)
        header = hdu.header
        this_hdu = cls(data, header)
        return this_hdu

    def to_image_hdu(self):
        return fits.ImageHDU(self.data, self.header)
    def to_primary_hdu(self):
        return fits.PrimaryHDU(self.data, self.header)

register_generic(DaskHDU)

class DaskHDUList:
    '''Represent a list of FITS header-data units in a way that is
    convenient for Dask to serialize and deserialize
    '''
    def __init__(self, hdus=None):
        if hdus is None:
            hdus = []
        self.hdus = hdus
    @classmethod
    def from_fits(cls, hdus):
        this_hdul = cls()
        for hdu in hdus:
            this_hdul.hdus.append(DaskHDU.from_fits(hdu))
        return this_hdul
    def to_fits(self):
        hdul = fits.HDUList([self.hdus[0].to_primary_hdu()])
        for imghdu in self.hdus[1:]:
            hdul.append(imghdu.to_image_hdu())
        return hdul
    def __contains__(self, key):
        if isinstance(key, int) and key < len(self.hdus):
            return True
        else:
            for hdu in self.hdus:
                if 'EXTNAME' in hdu.header and hdu.header['EXTNAME'] == key:
                    return True
        return False
    def __getitem__(self, key):
        for idx, hdu in enumerate(self.hdus):
            if 'EXTNAME' in hdu.header and hdu.header['EXTNAME'] == key:
                return hdu
            if idx == key:
                return hdu
        raise KeyError(f"No HDU {key}")

register_generic(DaskHDUList)

@dask.delayed
def load_fits(filename):
    log.debug(f'Loading {filename}')
    with open(filename, 'rb') as f:
        hdul = fits.open(f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log.debug(f'Validating {filename} FITS headers')
            hdul.verify('fix')
        log.debug(f'Converting {filename} to DaskHDUList')
        dask_hdul = DaskHDUList.from_fits(hdul)
    return dask_hdul

@dask.delayed
def write_fits(hdul, filename, overwrite=False):
    log.debug(f'Writing {filename}')
    hdul.to_fits().writeto(filename, overwrite=overwrite)
    return filename

@dask.delayed
def ensure_dq(hdul, like_ext=0):
    if 'DQ' in hdul:
        log.debug(f'Existing DQ extension found')
        return hdul
    log.info(hdul)
    log.info(hdul.hdus)
    dq_data = np.zeros_like(hdul[like_ext].data, dtype=np.uint8)
    dq_header = {
        'EXTNAME': 'DQ'
    }
    hdul.hdus.append(DaskHDU(dq_data, header=dq_header))
    log.debug(f'Added DQ extension based on extension {like_ext}')
    return hdul


@dask.delayed
def hdulists_to_dask_cube(all_hduls, ext=0):
    cube = da.stack([hdul[ext].data for hdul in all_hduls])
    return cube
