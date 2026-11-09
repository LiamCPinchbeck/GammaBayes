GammaBayes
==========

**GammaBayes** is a PyTorch library for forward-modelling binned gamma-ray data
(CTAO, H.E.S.S.-like IRFs) and performing Bayesian inference on it, with a focus on
dark-matter annihilation searches towards the Galactic Centre.

Version |release| is the *torch port*: every source component (spectral, spatial,
dark-matter spectra and J-factors, static templates) evaluates **batches** of parameter
values on the GPU and is differentiable, and the instrument response is applied with a
**count-conserving forward operator** from a padded *true* grid to the *reconstructed*
grid.  The older numpy-based analysis layer (event lists, parameter sets, nested-sampling
scans) still ships but is unmaintained; see :doc:`getting_started/migration`.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting_started/installation
   getting_started/concepts
   getting_started/migration

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/01_BinningGeometry
   tutorials/02_IRFs_and_Exposure
   tutorials/03_Source_Components
   tutorials/04_DarkMatter
   tutorials/05_IRF_Forward_Operator
   tutorials/06_Likelihood_Batching_Autograd
   tutorials/README

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
