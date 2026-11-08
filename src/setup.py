from setuptools import setup, find_packages
import subprocess, sys, os, time


# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(name='GammaBayes',
    description='A package for Bayesian dark matter inference on gamma-ray event data',
    url='https://github.com/LiamCPinchbeck/GammaBayes',
    author='Liam Pinchbeck',
    author_email='Liam.Pinchbeck@monash.edu',
    license="MIT",
    version='2.1.0',

    packages=find_packages(),
    long_description=long_description,  # This is the long description, read from README.md
    long_description_content_type="text/markdown",  
      # For a lot of the DM spectral classes we require that dict types are ordered
      python_requires='>=3.6, <=3.12.7',
      install_requires=[
        "numpy>=2.0",
        "astropy>=6.1",
        "gammapy>=2.0",
        "h5py>=3.11",
        "healpy>=1.17",
        "torch>=2.4",
        "scipy",
        "corner",
        "dynesty",
        "tqdm",
        "pyro-ppl",
        "pandas",
        "pytest",
        "icecream",
        "seaborn",
        "requests",
        "matplotlib",
    ],
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Operating System :: Unix",
],
      package_data={
          'gammabayes':['package_data/gll_iem_v06_gc.fits.gz', 
                        'package_data/hgps_catalog_v1.fits.gz',
                        'package_data/*.txt',
                        'package_data/*.fits',
                        'package_data/CTAO_irf_fits_files/*',
                        'package_data/CTAO_irf_fits_files/prod5/*.FITS.tar.gz',
                        'package_data/CTAO_irf_fits_files/prod3b/*',
                        'dark_matter/CosmiXs_Tables/*.dat',
                        'dark_matter/PPPC_Tables/*.dat',
                        'dark_matter/spectral_models/Z2_ScalarSinglet/annihilation_ratio_data/*',
                        'dark_matter/spectral_models/Z5/annihilation_ratio_data/*',
                        'standard_inference/*',
                        'utils/ozstar/*',
                        'utils/cli/*'
                        ]
      },
      )
