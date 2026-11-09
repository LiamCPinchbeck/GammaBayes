from setuptools import setup, find_packages


# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(name='GammaBayes',
      description='A package for Bayesian dark matter inference on gamma-ray event data',
      url='https://github.com/LiamCPinchbeck/GammaBayes',
      author='Liam Pinchbeck',
      author_email='Liam.Pinchbeck@monash.edu',
      license="MIT",
      version='2.1.1',

      packages=find_packages(),
      long_description=long_description,
      long_description_content_type="text/markdown",
      # 3.10+: the torch core uses PEP 604 unions (int | None) and dict-ordering guarantees
      python_requires='>=3.10',
      install_requires=[
          "numpy>=2.0",
          "scipy>=1.11",
          "astropy>=6.1",
          "gammapy>=2.0",
          "regions>=0.9,<0.11",   # regions >= 0.11 removed pixel_scale_angle_at_skycoord, which gammapy <= 2.1 imports
          "h5py>=3.11",
          "healpy>=1.17",
          "torch>=2.4",
          "pyro-ppl>=1.9",
          "pandas>=2.0",
          "matplotlib>=3.8",
          "tqdm",
          "icecream",
          "requests",
          "corner",
          "dynesty",
          "seaborn",
      ],
      extras_require={
          "test": ["pytest>=7"],
          "docs": ["sphinx>=7", "sphinx-rtd-theme>=2", "nbsphinx>=0.9", "myst-parser>=2",
                   "ipykernel", "nbconvert", "jupyter_client"],
      },
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          "Programming Language :: Python :: 3.12",
          "Programming Language :: Python :: 3.13",
          "Topic :: Scientific/Engineering :: Astronomy",
          "Topic :: Scientific/Engineering :: Physics",
      ],
      package_data={
          'gammabayes': ['package_data/gll_iem_v06_gc.fits.gz',
                         'package_data/hgps_catalog_v1.fits.gz',
                         'package_data/*.txt',
                         'package_data/*.fits',
                         'package_data/CTAO_irf_fits_files/*',
                         'package_data/CTAO_irf_fits_files/prod5/*.FITS.tar.gz',
                         'package_data/CTAO_irf_fits_files/prod3b/*',
                         'dark_matter/CosmiXs_Tables/*.dat',
                         'dark_matter/PPPC_Tables/*.dat',
                         'dark_matter/QCDUNC_Tables/*.dat',
                         'dark_matter/spectral_models/Z2_ScalarSinglet/annihilation_ratio_data/*',
                         'dark_matter/spectral_models/Z5/annihilation_ratio_data/*',
                         'utils/cli/*',
                         ]
      },
      )
