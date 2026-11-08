import torch
from functools import partial
from gammabayes.priors import TwoCompFluxPrior
from gammabayes.dark_matter import PPPCReader, SingleDMSpectralComp
from gammabayes.dark_matter.density_profiles import DM_Profile, Einasto_Profile
from gammabayes.core.batching import as_param


class SingleChannelDMPrior(TwoCompFluxPrior):

                                    # cm^3/s          TeV
    def __init__(self, channel='W', 
                    binning_geometry=None,
                    sigmav=3e-26, mass=1., symmetry_factor=torch.tensor(1.), 
                    spatial_profile=Einasto_Profile,
                    spatial_profile_kwargs=None,
                    **kwargs):


        self.mass = mass
        self.channel = channel
        self._single_channel_spectral_comp = SingleDMSpectralComp(
            binning_geometry=binning_geometry,
            channel=channel,
            mass=mass
        )

        if isinstance(spatial_profile, type):          # a profile CLASS -> instantiate (defaults if no kwargs)
            spatial_profile = spatial_profile(**(spatial_profile_kwargs or {}))

        self.spatial_comp = spatial_profile

        self.constant_spectral_prefactor = torch.tensor(1/(8*torch.pi))

        self.spectral_comp_logfunc = partial(self._spectral_comp_logfunc, 
                                                mass=mass, 
                                                sigmav=sigmav,
                                                symmetry_factor=symmetry_factor)

        super().__init__(
            binning_geometry=binning_geometry,
            spectral_comp = self.spectral_comp_logfunc,
            spatial_comp = self.spatial_comp,
            **kwargs
        )


    def _spectral_comp_logfunc(self, energy, mass, sigmav, symmetry_factor, kappa=None, delta=None):
        """Batched: mass / sigmav / symmetry_factor scalar or (B,) -> (B, E)."""
        E = torch.as_tensor(energy)
        mass_b = as_param(mass, 1, like=E)
        sigmav_b = as_param(sigmav, 1, like=E)
        sym_b = as_param(symmetry_factor, 1, like=E)
        logprefactor = torch.log(sigmav_b*self.constant_spectral_prefactor/(sym_b*mass_b**2))
        return logprefactor + self._single_channel_spectral_comp(E, mass=mass_b[:, 0], kappa=kappa, delta=delta)

