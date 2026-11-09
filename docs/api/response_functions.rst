Instrument response functions
=============================

.. automodule:: gammabayes.response_functions.core.irf_extractor_class

Kernels
-------

.. automodule:: gammabayes.response_functions.core.irf_kernel

.. warning::

   :func:`~gammabayes.response_functions.core.irf_kernel.make_irf_kernel` and
   :func:`~gammabayes.response_functions.core.irf_kernel.make_irf_kernel_per_energy` divide the
   energy width by :math:`\log_{10} e`, i.e. they interpret ``log10energy_std`` as if the kernel
   axis were in natural-log units while it is in :math:`\log_{10}`.  The effective resolution is
   therefore :math:`\ln 10 \approx 2.3` times the value passed (0.09 dex -> 0.207 dex).  This is
   kept for reproducibility of 0.x results; the forward operator
   (:func:`gammabayes.utils.irf_forward.make_forward_irf_kernels`) uses the width as given.

File helpers
------------

.. automodule:: gammabayes.response_functions.instrument_specific_interfaces.CTAO_IRFs.CTAO_irf_file_utils

.. automodule:: gammabayes.response_functions.core.irf_normalisation_setup
   :no-undoc-members:

.. automodule:: gammabayes.response_functions.normalisation_utils.fov_irf_slicer
   :no-undoc-members:
