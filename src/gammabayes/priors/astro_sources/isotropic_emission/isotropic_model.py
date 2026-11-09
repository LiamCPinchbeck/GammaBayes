"""Isotropic (sky-uniform) flux priors on the 2.x component API.

Both classes are :class:`TwoCompFluxPrior` instances with an ``IsotropicSpatial_PriorComp`` spatial part, so
``eval_log_on_geom(spectral_params={...})`` accepts batched spectral parameters exactly like the
other two-component priors.  ``phi0`` is the differential flux per deg^2 at ``ref_energy`` /
``break_energy`` in TeV^-1 cm^-2 s^-1 deg^-2 (the default 1e-14 is an illustrative diffuse level;
the 0.x default of 0.1 predates the per-deg^2 convention).
"""
import torch
from gammabayes.priors.core import TwoCompFluxPrior
from gammabayes.priors.spectral_components import BrokenPowerLaw, PowerLaw
from gammabayes.priors.spatial_components import IsotropicSpatial_PriorComp


class IsotropicPowerLaw(TwoCompFluxPrior):
    """phi0 (E/ref_energy)^-index, uniform on the sky."""

    def __init__(self, binning_geometry=None, index=2.5, phi0=1e-14, ref_energy=1., name='IsotropicGRB', **kwargs):
        spectral = PowerLaw(index=torch.as_tensor(index), phi0=torch.as_tensor(phi0), ref_energy=torch.as_tensor(ref_energy))
        super().__init__(spectral_comp=spectral, spatial_comp=IsotropicSpatial_PriorComp(),
                         binning_geometry=binning_geometry, name=name, **kwargs)


class IsotropicBrokenPowerLaw(TwoCompFluxPrior):
    """Broken power law (index1 below, index2 above ``break_energy`` [TeV]), uniform on the sky."""

    def __init__(self, binning_geometry=None, index1=2.3, index2=3.0, break_energy=0.3, phi0=1e-14,
                 name='IGRB', **kwargs):
        spectral = BrokenPowerLaw(index1=torch.as_tensor(index1), index2=torch.as_tensor(index2),
                                  break_energy=torch.as_tensor(break_energy), phi0=torch.as_tensor(phi0))
        super().__init__(spectral_comp=spectral, spatial_comp=IsotropicSpatial_PriorComp(),
                         binning_geometry=binning_geometry, name=name, **kwargs)
