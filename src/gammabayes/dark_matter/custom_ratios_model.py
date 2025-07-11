import numpy as np, copy
from os import path
ScalarSinglet_Folder_Path = path.dirname(__file__)
from collections.abc import Iterable   # import directly from collections for Python < 3.3
from collections import OrderedDict

from gammabayes.dark_matter.spectral_models.core import DM_ContinuousEmission_Spectrum, CSVDictionary
import time
from gammabayes.utils import logspace_simpson
from astropy import units as u

from gammabayes.dark_matter.channel_spectra import (
    single_channel_spectral_data_path,
    PPPCReader, 
)
import warnings
from gammabayes.dark_matter import CombineDMComps
from gammabayes.dark_matter.density_profiles import Einasto_Profile
from gammabayes.dark_matter.channel_spectra import (
    single_channel_spectral_data_path,
    
    PPPCReader, 
    SingleDMChannel
)
from gammabayes.likelihoods import DiscreteLogLikelihood

from tqdm import tqdm
from gammabayes.dark_matter.density_profiles import DM_Profile
from gammabayes import (
    GammaObs,
    ParameterSet, 
    apply_dirichlet_stick_breaking_direct, update_with_defaults
)

from scipy import special


class CustomDMRatiosModel(object):
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
                 irf_loglike:DiscreteLogLikelihood, 
                 spatial_class: DM_Profile = Einasto_Profile, 
                 spatial_class_kwds: dict= None,
                 channels: list[str] | str = 'high_mass',
                 default_spectral_parameters: dict = {},
                 default_spatial_parameters: dict = {},

                 ratios: dict = None,
                 *args, **kwargs
                 ):

        
        # Getting all possible channels as in the PPPC tables
        self.name = name
        all_channels = list(PPPCReader.darkSUSY_to_PPPC_converter.keys())
        self.is_single_channel = False

        
        if type(channels) == str:
            if channels.lower()=='all':
                # Ensuring that the first few elements are 
                    # 'W+W-', 'ZZ', 'HH', 'tt', 'bb', 'gg' in that order
                all_channels.remove('W+W-')
                all_channels.remove('HH')
                all_channels.remove('ZZ')
                all_channels.remove('tt')
                all_channels.remove('bb')
                all_channels.remove('gg')

                all_channels.insert(0, 'W+W-')
                all_channels.insert(1, 'ZZ')
                all_channels.insert(2, 'HH')
                all_channels.insert(3, 'tt')
                all_channels.insert(4, 'bb')
                all_channels.insert(5, 'gg')

                self.channels = all_channels
            elif channels in all_channels:

                warnings.warn("If specifying a single channel we recommend the use of the `SingleDMChannel` class in the gammabayes.dark_matter.channel_spectra module.")
                self.is_single_channel = True
                self.channels = [channels]
            else:
                raise ValueError(f"Invalid channel input given must be one or multiple of the following channels: {all_channels}")
        elif type(channels) == list:
            incorrect_channels = []

            for channel in channels:
                if channel not in all_channels:
                    incorrect_channels.append(channel)

            if len(incorrect_channels):
                raise ValueError(f"Invalid channels given ->{incorrect_channels} must be one or multiple of the following channels: {all_channels}")
            
            self.channels = channels


        self.channel_prior_dict = {}


        for channel in self.channels:

            if 'spectral_class_kwds' in kwargs:
                spectral_class_kwds = kwargs['spectral_class_kwds']
            else:
                spectral_class_kwds = {}

            spectral_class_kwds['channel'] = channel

            if spatial_class_kwds is not None:
                name=f"{channel} DM " + str(spatial_class_kwds) + " Class"
            else:
                name=f"{channel} DM Class"

            self.channel_prior_dict[channel] = CombineDMComps(name=name,
                        spectral_class = SingleDMChannel, 
                        spatial_class = spatial_class,
                        irf_loglike=irf_loglike, 
                        axes=axes, 
                        spectral_class_kwds = spectral_class_kwds,
                        spatial_class_kwds = spatial_class_kwds,
                        default_spectral_parameters=default_spectral_parameters,
                        default_spatial_parameters=default_spatial_parameters,
                        *args, **kwargs
                        )
                


    def generate_parameter_specifications(self, set_specifications_to_duplicate: dict = {}) -> dict:
        """
        Generates parameter specifications for all channels.

        Args:
            set_specifications_to_duplicate (dict, optional): Specifications to duplicate. Defaults to {}.

        Returns:
            dict: The parameter specifications for all channels.
        """

        input_set = set_specifications_to_duplicate

  
        all_parameter_sets = {}
        for channel in self.channels:
            all_parameter_sets[f'{channel} DM Class'] = ParameterSet(input_set)

        return all_parameter_sets



        
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
    

    def sample(self, 
               numevents: int| list = None,
               ratios:dict = None,
               ) -> dict[str, GammaObs]:
        """
        Samples events from the channel priors.

        Args:
            numevents (int | list): The number of events to sample. Defaults to None.

        Returns:
            dict[str, GammaObs]: The sampled event data.
        """
        
        if type(numevents) == int:
            if self.is_single_channel:
                return self.channel_prior_dict.values()[0].sample(numevents)
            
            event_data = {}
            for channel, channel_prior in self.channel_prior_dict.items():
                event_data[channel] = channel_prior.sample(numevents)

            return event_data
        

        
        elif isinstance(numevents, Iterable):
            if isinstance(numevents, dict):
                event_data = {}
                for channel, channel_prior in self.channel_prior_dict.items():
                    event_data[channel] = channel_prior.sample(numevents[channel])

            else:
                event_data = {}
                for channel_idx, [channel, channel_prior] in enumerate(self.channel_prior_dict.items()):
                    event_data[channel] = channel_prior.sample(numevents[channel_idx])

            return event_data
        

        else:
            raise ValueError("""'numevents' must be an int or some iterably 
(preferably a dictionary with keys being the channels or list in the same order as specified channels).""")
        

    def sample_from_weights(self, 
                            num_events: int, 
                            input_weights: list[float] | dict[float], 
                            stick_breaking: bool = False, 
                            exhaustive_fractions=False) -> dict[str, GammaObs]:
        """
        Samples events from the channel priors based on input weights.

        Args:
            num_events (int): The total number of events to sample.
            input_weights (list[float] | dict[float]): The weights for each channel.
            stick_breaking (bool, optional): Whether to use stick-breaking for the weights. Defaults to False.
            exhaustive_fractions (bool, optional): Whether the weights are exhaustive fractions. Defaults to False.

        Returns:
            dict[str, GammaObs]: The sampled event data.
        """
        


        
        if type(input_weights) == dict:
            if exhaustive_fractions:
                input_weights = [input_weights[key] for key in self.channels[:-1]]
            else:
                input_weights = [input_weights[key] for key in self.channels]
        
        if stick_breaking:
            formatted_weights = []
            for depth, channel in enumerate(self.channels):
                formatted_weight = apply_dirichlet_stick_breaking_direct(input_weights, depth) 

                formatted_weights.append(formatted_weight)

        else:
            formatted_weights = input_weights

        return {channel: prior.sample(int(round(weight*num_events))) for weight, [channel, prior] in zip(formatted_weights, self.channel_prior_dict.items())}
    



    def convert_sample_param_to_sigmav(self,
                            overall_signal_fraction, 
                            ratios:dict = None,
                            true_axes=None,
                            spectral_parameters = {},
                            spatial_parameters = {},
                            totalnumevents=1e8, 
                            tobs_seconds=525*60*60, 
                            symmetryfactor=1, chunk_size=2):
        """
        Converts samples of dark matter parameters to the annihilation/decay cross-section.

        Presumes that parameter values are given in 1d arrays or lists.

        Args:
            signal_fraction (float): The fraction of the signal.
            true_axes (list, optional): The true axes values. Defaults to None.
            spectral_parameters (dict, optional): Spectral parameters. Defaults to {}.
            spatial_parameters (dict, optional): Spatial parameters. Defaults to {}.
            totalnumevents (float, optional): The total number of events. Defaults to 1e8.
            tobs_seconds (float, optional): The observation time in seconds. Defaults to 525*60*60.
            symmetryfactor (int, optional): The symmetry factor. Defaults to 1.

        Returns:
            np.ndarray: The annihilation cross-section values.
        """
        
        # update_with_defaults(spectral_parameters, self.default_spectral_parameters)
        # update_with_defaults(spatial_parameters, self.default_spatial_parameters)

        

        if true_axes is None:
            true_axes = self.axes

        all_parameters = list(spectral_parameters.values()) + list(spatial_parameters.values())

        num_spectral_params = len(spectral_parameters.keys())
        parameter_values = np.asarray([item.value if hasattr(item, "value") else item for item in all_parameters])

        unique_arrays_of_parameter_values = np.unique(parameter_values, axis=1)

        unique_dict_of_spec_parameter_values = {key:unique_arrays_of_parameter_values[key_idx] for key_idx, key in enumerate(spectral_parameters.keys())}
        unique_dict_of_spat_parameter_values = {key:unique_arrays_of_parameter_values[num_spectral_params+key_idx] for key_idx, key in enumerate(spatial_parameters.keys())}


        axis_mesh_shape = (*(len(axis) for axis in true_axes), len(unique_arrays_of_parameter_values[0]))

        individual_integrands = {}


        for channel, channel_prior in self.channel_prior_dict.items():
            channel_prior.log_scaling_factor = 0
            individual_integrands[channel] = channel_prior.construct_prior_array(
                                                    spectral_parameters=unique_dict_of_spec_parameter_values,
                                                    spatial_parameters=unique_dict_of_spat_parameter_values).reshape(axis_mesh_shape)
        
        # Splitting it up for easier debuggin
        log_integrated_energies = {channel: logspace_simpson(
                                    logy=integrand, x = true_axes[0].value, axis=0) for channel, integrand in individual_integrands.items()}

        log_integrated_energy_longitudes = {channel: logspace_simpson(
                                    logy=integrand, x = true_axes[1].value, axis=0) for channel, integrand in log_integrated_energies.items()}

        mask = np.all(unique_arrays_of_parameter_values[:, None, :] == parameter_values[:, :, None], axis=0)

        log_integrals = {}
        for channel, integrand in log_integrated_energy_longitudes.items():
            log_channel_unique_integral = logspace_simpson(
                                logy=integrand.T+np.log(np.sin(np.abs(true_axes[2].to(u.rad))))+np.log(180/np.pi), 
                                        x = true_axes[2].value, axis=-1)
            

            log_channel_integral_slices = np.where(mask, log_channel_unique_integral.flatten()[None, :], 0.)
        
            log_integrals[channel] = np.sum(log_channel_integral_slices, axis=1)



        logintegral = special.logsumexp([np.log(ratios[channel])+log_integral for channel, log_integral in log_integrals.items()], axis=0)


        logsigmav = np.log(8*np.pi*symmetryfactor*spectral_parameters['mass'].value**2*totalnumevents*overall_signal_fraction) - logintegral - np.log(tobs_seconds)

        return np.exp(logsigmav)





        

        





