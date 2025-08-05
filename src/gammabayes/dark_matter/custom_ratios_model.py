import torch
from functools import partial
from gammabayes.priors import TwoCompFluxPrior
from gammabayes.dark_matter import PPPCReader, SingleDMSpectralComp
from gammabayes.dark_matter.density_profiles import DM_Profile, Einasto_Profile



class CustomDMRatiosModel(TwoCompFluxPrior):
    """
    A class to model custom dark matter (DM) ratios for different channels.

    Attributes:
        irf_loglike (DiscreteLogLikelihood): The likelihood function based on instrument response functions (IRFs).
        axes (list | tuple | np.ndarray): The axes for the likelihood function.
        spatial_class (DM_Profile): The spatial profile class for dark matter density.
        channels (list[str] | str): The list of channels or 'all' for all available channels.
        default_spectral_parameters (dict): Default spectral parameters.
        default_spatial_parameters (dict): Default spatial parameters.
    """

    def __init__(self, 
                    binning_geometry=None,
                    mass = torch.tensor(1.),
                    sigmav=3e-26, 
                    spatial_profile=Einasto_Profile,
                    spatial_profile_kwargs=None,
                    channels = ['WW', 'ZZ', 'HH', 'tt'],
                    ratios = None,
                    symmetry_factor=torch.tensor(1.), 
                    *args, **kwargs
                 ):

        
        all_channels = list(PPPCReader.darkSUSY_to_PPPC_converter.keys())
        self.is_single_channel = False

        
        if type(channels) == list:
            incorrect_channels = []

            for channel in channels:
                if channel not in all_channels:
                    incorrect_channels.append(channel)

            if len(incorrect_channels):
                raise ValueError(f"Invalid channels given ->{incorrect_channels} must be one or multiple of the following channels: {all_channels}")
            
            self.channels = channels

        else:
            raise ValueError("Currently require that the channels input is a list of strings for the channels")

        print(self.channels)


        self.spectral_models = {}
        # *** For each channel instantiate the spectral component and store it in a dict ***
        for channel in self.channels:
            self.spectral_models[channel] = SingleDMSpectralComp(
                binning_geometry=binning_geometry, channel=channel, mass=mass,
            )

        self.constant_spectral_prefactor = torch.tensor(1/(8*torch.pi))

        self.spectral_comp_logfunc = partial(self._spectral_comp_logfunc, 
                                                mass=mass, 
                                                sigmav=sigmav,
                                                ratios=ratios,
                                                symmetry_factor=symmetry_factor)
        if spatial_profile_kwargs is not None:
            spatial_profile = spatial_profile(**spatial_profile_kwargs)

        self.spatial_comp = spatial_profile


        super().__init__(
            binning_geometry=binning_geometry,
            spectral_comp = self.spectral_comp_logfunc,
            spatial_comp = self.spatial_comp,
            **kwargs
        )



    def _multi_channel_spectral_comp(self, energy, mass, ratios):

        log_output_specs = []
        for channel, ratio in ratios.items():
            log_output_specs.append(torch.log(ratio) + self.spectral_models[channel](energy, mass=mass))

        log_output_spec = torch.logsumexp(torch.stack(log_output_specs, dim=0), dim=0)

        return log_output_spec



    def _spectral_comp_logfunc(self, energy, mass, ratios, sigmav, symmetry_factor):

        logprefactor = torch.log(sigmav*self.constant_spectral_prefactor/(symmetry_factor*mass**2))

        mass = mass*torch.ones_like(energy)

        return logprefactor + self._multi_channel_spectral_comp(energy, mass=mass, ratios=ratios)






        
    def __iter__(self):
        """
        Returns an iterator for the channel prior dictionary.

        Returns:
            iterator: An iterator for the channel prior dictionary.
        """
        return self.channel_prior_dict.__iter__()
    
    def items(self):
        """
        Returns the items of the channel prior dictionary.

        Returns:
            dict_items: The items of the channel prior dictionary.
        """
        return self.channel_prior_dict.items()
    
    def keys(self):
        """
        Returns the keys of the channel prior dictionary.

        Returns:
            dict_keys: The keys of the channel prior dictionary.
        """
        return self.channel_prior_dict.keys()
    
    def values(self):
        """
        Returns the values of the channel prior dictionary.

        Returns:
            dict_values: The values of the channel prior dictionary.
        """
        return self.channel_prior_dict.values()
    




        

        





