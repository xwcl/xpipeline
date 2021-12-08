#!/usr/bin/env python3
from setuptools import setup, find_packages

description = 'The eXtreme (and eXtensible) pipeline for analysis of high contrast imaging and spectroscopy data'

setup(
    name='xpipeline',
    # version='0.0.1.dev',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    url='https://github.com/magao-x/xpipeline',
    description=description,
    author='Joseph D. Long',
    author_email='me@joseph-long.com',
    packages=find_packages(),
    package_data={
        'xpipeline.ref': [
            '3.9um_Clio.dat',
            'naco_betapic_preproc_absil2013_gonzalez2017.npz'
        ],
        'xpipeline.ref.clio': [
            '3.9um_Clio.dat',
            'vAPP_transmission.fits',
            'vAPP_phase_top.fits',
            'vAPP_phase_bottom.fits',
            'clio_pupil_mask.fits',
        ],
    },
    install_requires=[
        'pytest>=6.2.1',
        'numpy>=1.20.3',  # first version with period= argument to unwrap
        'scipy>=1.4.1',
        'matplotlib>=3.2.2',
        'astropy>=4.2',
        'dask>=2021.1.1',
        'distributed>=2021.1.1',
        'python-irodsclient>=0.8.6',
        'coloredlogs>=15.0',
        'scikit-image>=0.18.1',
        'irods_fsspec',
        'fsspec>=0.8.7',
        'sphinx-argparse>=0.2.5',
        'sphinx-rtd-theme>=0.4.3',
        'python-dateutil>=2.8.1,<3',
        'dacite>=1.6.0,<2.0',
        'numba>=0.53,<1.0',
        'memory-profiler>=0.58.0',
        'ray>=1.8',
        'pylops>=1.14,<2.0',
        'tqdm>=4.62',
        'pandas>=1.3.4',
    ],
    entry_points={
        'console_scripts': [
            'xp=xpipeline.cli:main',
            'xp_local_to_irods=xpipeline.commands:local_to_irods',
            'xp_compute_sky_model=xpipeline.commands:compute_sky_model',
            'xp_clio_instrument_calibrate=xpipeline.commands:clio_instrument_calibrate'
        ],
    }
)
