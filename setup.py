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
        "corner==2.2.2",
        "dynesty==2.1.2",
        "tqdm==4.66.1",
        "gammapy==1.2",
        "pandas==2.1.2",
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
                        'dark_matter/channel_spectra/PPPC_Tables/*.dat',
                        'dark_matter/spectral_models/Z2_ScalarSinglet/annihilation_ratio_data/*',
                        'dark_matter/spectral_models/Z5/annihilation_ratio_data/*',
                        'standard_inference/*',
                        'utils/ozstar/*',
                        'utils/cli/*'
                        ]
      },
      entry_points={
          'console_scripts':[
            'GammaBayes = gammabayes.utils.cli.intro:CLI_Intro',
            'gammabayes = gammabayes.utils.cli.intro:CLI_Intro',
            'gammabayes.slurm.run_initial_setup = gammabayes.utils.cli.slurm:initial_setup',
            'gammabayes.slurm.run_simulate = gammabayes.utils.cli:run_sim',
            'gammabayes.slurm.run_analysis_setup = gammabayes.utils.cli.slurm:analysis_setup',
            'gammabayes.slurm.run_marg = gammabayes.utils.cli:run_marg',
            'gammabayes.slurm.run_combine = gammabayes.utils.cli:run_combine',
            'gammabayes.run_simulate = gammabayes.utils.cli:run_sim',
            'gammabayes.run_marg = gammabayes.utils.cli:run_marg',
            'gammabayes.run_combine = gammabayes.utils.cli:run_combine',
            'gammabayes.plot_from_save = gammabayes.utils.cli:plot_from_save',
            ]
      },

      )
