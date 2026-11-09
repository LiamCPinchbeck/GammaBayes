Dark matter
===========

The DM annihilation model is a :class:`~gammabayes.priors.core.two_comp_prior.TwoCompFluxPrior`
whose spectral part is

.. math::

   \frac{d\phi}{dE} = \frac{\langle\sigma v\rangle}{8\pi\, S_\chi\, m_\chi^2}
                      \sum_{c} B_c \frac{dN_c}{dE}(E;\, m_\chi)

(``S_\chi`` the symmetry factor, 1 for Majorana) and whose spatial part is the differential
J-factor :math:`dJ/d\Omega = \int \rho^2\, ds` of a density profile.

High-level models
-----------------

.. automodule:: gammabayes.dark_matter.custom_ratios_model

.. automodule:: gammabayes.dark_matter.single_channel_prior

Density profiles and J-factors
------------------------------

.. automodule:: gammabayes.dark_matter.density_profiles.base_dm_profile

.. automodule:: gammabayes.dark_matter.density_profiles.DM_Profiles

Photon spectra per channel
--------------------------

.. automodule:: gammabayes.dark_matter.PPPC_Tables.single_channel_DM_Spectral_Class

Table readers
^^^^^^^^^^^^^

.. automodule:: gammabayes.dark_matter.PPPC_Tables.PPPC_reader
   :members: PPPCReader
   :no-undoc-members:

.. automodule:: gammabayes.dark_matter.PPPC_Tables.cosmixs_reader

.. automodule:: gammabayes.dark_matter.PPPC_Tables.qcdunc_reader

Model-specific spectra (legacy numpy)
-------------------------------------

These classes predate the torch port and produce numpy spectra for the scalar-singlet and
:math:`Z_5` models.  They are usable on their own but are not wired into the batched priors.

.. automodule:: gammabayes.dark_matter.spectral_models.core.csv_to_dict_class
   :no-undoc-members:

.. automodule:: gammabayes.dark_matter.spectral_models.core.base_spectral_class
   :no-undoc-members:

.. automodule:: gammabayes.dark_matter.spectral_models.core.base_multi_DM_spectral_class
   :no-undoc-members:

.. automodule:: gammabayes.dark_matter.spectral_models.Z2_ScalarSinglet.SS_DM_Spectral_Class
   :no-undoc-members:

.. automodule:: gammabayes.dark_matter.spectral_models.Z5.Z5_spectral_class
   :no-undoc-members:
