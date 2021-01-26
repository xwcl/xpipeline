'''Loading and saving FITS with delayed functions and simplified data structures

Astropy FITS support creates objects with memory-mapped arrays and locks,
neither of which can be shipped over the network for distributed execution.
This module reads `astropy.io.fits.HDUList` objects into `DaskHDUList` 
objects, converting `numpy.ndarray` data arrays to `dask.array` data arrays
(but reusing `astropy.io.fits.header.Header`)
'''
from .. import irods
from distributed.protocol import register_generic
import numpy as np
from astropy.io import fits
import dask.array as da
import dask
import logging
import os.path
import warnings
log = logging.getLogger(__name__)


class DaskHDU:
    '''Represent a FITS header-data unit in a way that is
    convenient for Dask to serialize and deserialize
    '''

    def __init__(self, data, header=None):
        if header is None:
            header = {}
        self.data = data
        self.header = fits.Header(header)

    @classmethod
    def from_fits(cls, hdu, distributed=False):
        if distributed:
            data = da.from_array(hdu.data)
        else:
            data = hdu.data.copy()
        header = hdu.header
        this_hdu = cls(data, header)
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

    def to_image_hdu(self):
        return fits.ImageHDU(self.data, self.header)

    def to_primary_hdu(self):
        return fits.PrimaryHDU(self.data, self.header)


register_generic(DaskHDU)


def _check_ext(key, hdu, idx):
    if 'EXTNAME' in hdu.header and hdu.header['EXTNAME'] == key:
        return True
    if idx == key:
        return True
    return False


class DaskHDUList:
    '''Represent a list of FITS header-data units in a way that is
    convenient for Dask to serialize and deserialize
    '''

    def __init__(self, hdus=None):
        if hdus is None:
            hdus = []
        self.hdus = hdus

    @classmethod
    def from_fits(cls, hdus, distributed=False):
        this_hdul = cls()
        for hdu in hdus:
            this_hdul.hdus.append(DaskHDU.from_fits(hdu, distributed=distributed))
        return this_hdul

    def to_fits(self):
        hdul = fits.HDUList([self.hdus[0].to_primary_hdu()])
        for imghdu in self.hdus[1:]:
            hdul.append(imghdu.to_image_hdu())
        return hdul

    def copy(self):
        return self.__class__([hdu.copy() for hdu in self.hdus])

    def updated_copy(self, new_data, new_headers=None, history=None, ext=0):
        '''Return a copy of the DaskHDUList with extension `ext`'s data
        replaced by `new_data`, optionally appending a history card
        with what update was performed
        '''
        new_hdus = []
        for idx, hdu in enumerate(self.hdus):
            if _check_ext(ext, hdu, idx):
                new_hdu = hdu.updated_copy(new_data, new_headers=new_headers, history=history)
                new_hdus.append(new_hdu)
            else:
                new_hdus.append(hdu)
        return self.__class__(new_hdus)

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
            if _check_ext(key, hdu, idx):
                return hdu
        raise KeyError(f"No HDU {key}")


register_generic(DaskHDUList)


@dask.delayed
def load_fits_from_disk(filename):
    log.debug(f'Loading {filename}')
    with open(filename, 'rb') as f:
        hdul = fits.open(f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log.debug(f'Validating {filename} FITS headers')
            hdul.verify('fix')
            for hdu in hdul:
                hdu.header.add_history('xpipeline loaded and validated format')
        log.debug(f'Converting {filename} to DaskHDUList')
        dask_hdul = DaskHDUList.from_fits(hdul)
    return dask_hdul


def get_data_from_disk(filename, ext=0):
    return load_fits_from_disk(filename)[ext].data


@dask.delayed
def write_fits_to_disk(hdul, filename, overwrite=False):
    log.debug(f'Writing {filename}')
    hdul.to_fits().writeto(filename, overwrite=overwrite)
    return filename


def flatten_single_header(hdu, prefix):
    return {
        f'{prefix}.{key}': val
        for key, val
        in hdu.header.items()
    }


def flatten_headers(hdul):
    primary_hdu, rest_hdus = hdul.hdus[0], hdul.hdus[1:]
    output = flatten_single_header(primary_hdu, 'fits')
    for idx, hdu in enumerate(rest_hdus, start=1):
        extname = hdu.header.get('EXTNAME', f'ext{idx}')
        prefix = f'fits.{extname}'
        output.update(flatten_single_header(hdu, prefix))
    return output


@dask.delayed
def write_fits_to_irods(hdul, destination_path, overwrite=False):
    log.debug(f'Writing to {destination_path} on iRODS')
    session = irods.get_session()
    collection_path, data_object_name = os.path.split(destination_path)
    data_object_exists = session.data_objects.exists(destination_path)
    if not overwrite and data_object_exists:
        raise RuntimeError(f"Found existing iRODS data object at {destination_path}")
    if not data_object_exists:
        log.debug(f'Creating iRODS data object {destination_path}')
        destfile = session.data_objects.create(destination_path)
    else:
        log.debug(f'Reusing existing iRODS data object {destination_path}')
        destfile = session.data_objects.get(destination_path)
    
    with destfile.open(mode='w') as destfh:
        log.info(f'Writing to iRODS data object at {destination_path} ({destfile})')
        hdul.to_fits().writeto(destfh)
    irods.attach_metadata(
        destfile,
        flatten_headers(hdul)
    )
    return destination_path


@dask.delayed
def ensure_dq(hdul, like_ext=0):
    if 'DQ' in hdul:
        log.debug('Existing DQ extension found')
        return hdul
    dq_data = np.zeros_like(hdul[like_ext].data, dtype=np.uint8)
    dq_header = {
        'EXTNAME': 'DQ'
    }
    hdul.hdus.append(DaskHDU(dq_data, header=dq_header))
    msg = f'Created DQ extension based on extension {like_ext}'
    log.info(msg)
    hdul['DQ'].header.add_history(msg)
    return hdul


@dask.delayed
def hdulists_to_dask_cube(all_hduls, ext=0):
    cube = da.stack([
        hdul[ext].data
        for hdul in all_hduls
    ])
    log.info('Inputs stacked into cube')
    return cube
