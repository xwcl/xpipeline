#!/usr/bin/env python3
from setuptools import setup, find_packages

description = 'The eXtreme (and eXtensible) pipeline for analysis of high contrast imaging and spectroscopy data'

setup(
    name='xpipeline',
    version='0.0.1.dev',
    url='https://github.com/magao-x/xpipeline',
    description=description,
    author='Joseph D. Long',
    author_email='me@joseph-long.com',
    packages=['xpipeline'],
    # package_data={
    #     'doodads.ref': ['3.9um_Clio.dat'],
    # },
    install_requires=['pytest>=5.4.2', 'numpy>=1.18.4', 'scipy>=1.2.1',
                      'matplotlib>=3.1.3', 'astropy>=4.0.1', 'dask>=2.30.0'],
    entry_points={
        'console_scripts': [
            'xp_ingest=xpipeline.commands:ingest',
            'xp_compute_sky_model=xpipeline.commands:compute_sky_model',
            'xp_clio_instrument_calibrate=xpipeline.commands:clio_instrument_calibrate'
        ],
    }
)
