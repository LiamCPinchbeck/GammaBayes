"""
custom_dm_ratios_model_v2.py — REPLACES the gammabayes module defining CustomDMRatiosModel.
Only change vs your current file: kappa is threaded through the spectral chain
(_spectral_comp_logfunc -> _multi_channel_spectral_comp -> SingleDMSpectralComp),
so the driver can pass spectral_params={"mass":..., "sigmav":..., "ratios":..., "kappa": k}.
kappa=None (default) reproduces current behavior exactly.
"""
import torch
from functools import partial
from gammabayes.priors import TwoCompFluxPrior
from gammabayes.dark_matter import PPPCReader, SingleDMSpectralComp
from gammabayes.dark_matter.density_profiles import DM_Profile, Einasto_Profile


class CustomDMRatiosModel(TwoCompFluxPrior):
    """Custom DM channel-ratio model; v2 = kappa spectral nuisance threaded through."""

    def __init__(self,
                 binning_geometry=None,
                 mass=torch.tensor(1.),
                 sigmav=3e-26,
                 spatial_profile=Einasto_Profile,
                 spatial_profile_kwargs=None,
                 channels=['WW', 'ZZ', 'HH', 'tt'],
                 ratios=None,
                 symmetry_factor=torch.tensor(1.),
                 spectral_reader='cosmixs',
                 *args, **kwargs):

        all_channels = list(PPPCReader.darkSUSY_to_PPPC_converter.keys())
        self.is_single_channel = False

        if type(channels) == list:
            incorrect_channels = [ch for ch in channels if ch not in all_channels]
            if len(incorrect_channels):
                raise ValueError(f"Invalid channels given ->{incorrect_channels} must be one or "
                                 f"multiple of the following channels: {all_channels}")
            self.channels = channels
        else:
            raise ValueError("Currently require that the channels input is a list of strings for the channels")
        print(self.channels)

        self.spectral_models = {}
        for channel in self.channels:
            self.spectral_models[channel] = SingleDMSpectralComp(
                binning_geometry=binning_geometry, channel=channel, mass=mass,
                reader=spectral_reader)

        self.constant_spectral_prefactor = torch.tensor(1/(8*torch.pi))

        self.spectral_comp_logfunc = partial(self._spectral_comp_logfunc,
                                             mass=mass, sigmav=sigmav, ratios=ratios,
                                             symmetry_factor=symmetry_factor)
        if spatial_profile_kwargs is not None:
            spatial_profile = spatial_profile(**spatial_profile_kwargs)
        self.spatial_comp = spatial_profile

        super().__init__(
            binning_geometry=binning_geometry,
            spectral_comp=self.spectral_comp_logfunc,
            spatial_comp=self.spatial_comp,
            **kwargs)

    def _multi_channel_spectral_comp(self, energy, mass, ratios, kappa=None, delta=None):
        log_output_specs = []
        for channel, ratio in ratios.items():
            log_output_specs.append(torch.log(ratio)
                                    + self.spectral_models[channel](energy, mass=mass, kappa=kappa, delta=delta))
        return torch.logsumexp(torch.stack(log_output_specs, dim=0), dim=0)

    def _spectral_comp_logfunc(self, energy, mass, ratios, sigmav, symmetry_factor, kappa=None, delta=None):
        logprefactor = torch.log(sigmav*self.constant_spectral_prefactor/(symmetry_factor*mass**2))
        mass = mass*torch.ones_like(energy)
        return logprefactor + self._multi_channel_spectral_comp(energy, mass=mass, ratios=ratios, kappa=kappa, delta=delta)

    def __iter__(self):
        return self.channel_prior_dict.__iter__()

    def items(self):
        return self.channel_prior_dict.items()

    def keys(self):
        return self.channel_prior_dict.keys()

    def values(self):
        return self.channel_prior_dict.values()