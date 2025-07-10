from setuptools import setup, find_packages
import subprocess, sys, os, time


# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(name='GammaBayes',
      description='A package for Bayesian dark matter inference on gamma-ray event data',
      url='https://github.com/lpin0002/GammaBayes',
      author='Liam Pinchbeck',
      author_email='Liam.Pinchbeck@monash.edu',
      license="MIT",
      version='2.0.0',

      packages=find_packages(),
        long_description=long_description,  # This is the long description, read from README.md
    long_description_content_type="text/markdown",  
      # For a lot of the DM spectral classes we require that dict types are ordered
      python_requires='>=3.6, <=3.12.7',
      install_requires=[
        "astropy==5.3.4",
        "corner",
        "dynesty==2.1.2",
        "tqdm",
        "gammapy==1.2",
        "pandas",
        "pytest==7.4.0",
        "h5py==3.10.0",
        "icecream==2.1.3",
        "seaborn==0.12.2",
        "requests==2.31.0",
        "matplotlib==3.9.2",
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
                        'dark_matter/PPPC_Tables/*.dat',
                        'dark_matter/spectral_models/Z2_ScalarSinglet/annihilation_ratio_data/*',
                        'dark_matter/spectral_models/Z5/annihilation_ratio_data/*',
                        'standard_inference/*',
                        'utils/ozstar/*',
                        'utils/cli/*'
                        ]
      },
      )
