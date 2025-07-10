import numpy as np
from scipy import interpolate
from os import path
ScalarSinglet_Folder_Path = path.dirname(__file__)

from gammabayes.dark_matter.channel_spectra import (
    single_channel_spectral_data_path,
    PPPCReader, 
)
from astropy import units as u
from gammabayes import update_with_defaults



single_Wchannel_annihilation_ratios   = {}
mass_axis       = np.logspace(-1,2,301)

for channel in PPPCReader.darkSUSY_to_PPPC_converter:
    if channel[0]=="W":
        single_Wchannel_annihilation_ratios[channel] = mass_axis*0+1
    else:
        single_Wchannel_annihilation_ratios[channel] = mass_axis*0


class DM_ContinuousEmission_Spectrum(object):
    """
    Class to compute the continuous emission spectrum from dark matter annihilation across various channels.

    Attributes:
        ratios (bool): Indicates if the cross-sections should be interpreted as ratios.
        sqrtchannelfuncdictionary (dict): Dictionary of interpolated functions for each darkSUSY channel spectrum (needs to be squared).
        partial_sqrt_sigmav_interpolator_dictionary (dict): Dictionary of interpolated functions for annihilation ratios (needs to be squared).
        parameter_interpolation_values (list): List of parameter values for interpolation.
        parameter_axes (list): List of unique parameter values.
        annihilation_fractions (dict): Dictionary of annihilation fractions.
        default_parameter_values (dict): Default values for parameters.
    """


    def zero_output(self, inputval):
        """
        A helper method that returns a zero value for any given input.

        Args:
            inputval: Input value or array.

        Returns:
            The zero value of the same shape as the input.
        """
        return inputval[0]*0
    def one_output(self, inputval):
        """
        A helper method that returns a one value for any given input.

        Args:
            inputval: Input value or array.

        Returns:
            A value or array of ones of the same shape as the input.
        """
        return inputval[0]*0 + 1
    
    def __init__(self, 
                 annihilation_fractions= None, 
                 parameter_interpolation_values:  list[np.ndarray] = None, 
                 ratios: bool = True,
                 default_parameter_values = None,
                 ):
        """
        Initialize the DMContinuousEmissionSpectrum class.

        Args:
            annihilation_fractions (dict): Dictionary mapping annihilation channels to their fractions.
            parameter_interpolation_values (list[np.ndarray]): List of parameter values for interpolation of annihilation_fractions.
            ratios (bool, optional): Indicates if the fractions are ratios. Defaults to True.
            default_parameter_values (dict, optional): Default values for parameters. Defaults to {'mass': 1.0}.
        """
        if annihilation_fractions is None:
            annihilation_fractions = single_Wchannel_annihilation_ratios

        if parameter_interpolation_values is None:
            parameter_interpolation_values = [mass_axis]

        if default_parameter_values is None:
            default_parameter_values = {'mass':1.0,}


        
        np.seterr(divide='ignore')
        self.ratios = ratios
    
        # This class presumes that you're getting your annihilation ratios from darkSUSY
            # If you're using something else (e.g. MicroOMEGAs) you will need to check that this 
            # dictionary is correct for your case


        atprod_gammas = PPPCReader(single_channel_spectral_data_path+"/PPPC_Tables/AtProduction_gamma_EW_corrections.dat")
        atprod_mass_values = atprod_gammas.mass_axis
        atprod_log10x_values = atprod_gammas.log10x_axis

        # We take the square root of the outputs so later we can square them to enforce positivity
        sqrtchannelfuncdictionary = {}
        for darkSUSYchannel in PPPCReader.darkSUSY_to_PPPC_converter.keys():
            try:
                PPPC_channel = PPPCReader.darkSUSY_to_PPPC_converter[darkSUSYchannel]
                
                # Extracting single channel spectra
                tempspectragrid = atprod_gammas[PPPC_channel].reshape(atprod_gammas.output_shape)
                
                # Interpolating square root of PPPC tables to preserve positivity during interpolation (where result is squared)
                sqrtchannelfuncdictionary[darkSUSYchannel] = interpolate.RegularGridInterpolator(
                    (np.log10(atprod_mass_values), atprod_log10x_values), 
                    np.sqrt(np.asarray(tempspectragrid)),
                    method='cubic', bounds_error=False, fill_value=0)
            except:
                sqrtchannelfuncdictionary[darkSUSYchannel] = self.zero_output # inputs should be a tuple or list of log_10(mass) in TeV and log_10(x)

        # Saving result to class
        self.sqrtchannelfuncdictionary = sqrtchannelfuncdictionary


        if self.ratios:
            self.log_normalisations = -np.inf
            for channel in annihilation_fractions.keys():
                self.log_normalisations = np.logaddexp(self.log_normalisations, np.log(annihilation_fractions[channel]))

            self.log_normalisations = np.where(np.isfinite(self.log_normalisations), self.log_normalisations, 0)

            for channel in annihilation_fractions.keys():
                new_annihilation_fractions = np.exp(np.log(annihilation_fractions[channel])-self.log_normalisations)
                annihilation_fractions[channel] = new_annihilation_fractions


        
        
        # sqrt enforces positivity while also transforming values to closer to 1
        if len(parameter_interpolation_values)>1:
            self.partial_sqrt_sigmav_interpolator_dictionary = {
                channel: interpolate.RegularGridInterpolator(
                    (*parameter_interpolation_values,),
                    np.sqrt(annihilation_fractions[channel]),
                    # 'cubic' method can be unstable for small values, we highly recommend 
                        # you use ratios=True unless otherwise required
                    method='cubic', bounds_error=False, fill_value=0) \
                        for channel in list(PPPCReader.darkSUSY_to_PPPC_converter.keys()
                    )
                    }
        else:
            self.partial_sqrt_sigmav_interpolator_dictionary = {
                channel: interpolate.interp1d(
                    parameter_interpolation_values[0],
                    np.sqrt(annihilation_fractions[channel]),
                    kind='cubic', bounds_error=False, fill_value='extrapolate') for channel in list(PPPCReader.darkSUSY_to_PPPC_converter.keys()
                    )
                    }
            
        self.parameter_interpolation_values = parameter_interpolation_values
        self.parameter_axes = [np.unique(values) for values in self.parameter_interpolation_values]
        self.annihilation_fractions = annihilation_fractions
        self.default_parameter_values = default_parameter_values
        self.inverse_output_unit = u.TeV



    
    def __call__(self, *args, **kwargs) -> np.ndarray | float:
        """
        Allows the instance to be called as a function, delegating to the `logfunc` method.

        Args and Returns:
            See `logfunc` method for detailed argument and return value descriptions.
        """
        return self.logfunc(*args, **kwargs)


    def spectral_gen(self, energy: float | np.ndarray | list, 
                           **kwargs) -> np.ndarray | float:
        """
        Generates the dark matter annihilation gamma-ray spectrum.

        Args:
            energy (float | np.ndarray | list): Gamma-ray energies in TeV.
            **kwargs: Additional keyword arguments for model parameters.

        Returns:
            np.ndarray | float: Gamma-ray flux for the specified energies and parameters.
        """
        

        logspectra = -np.inf



        update_with_defaults(kwargs, self.default_parameter_values)


        try:
            energy = energy.to("TeV").value
        except:
            pass

        try:
            mass_value = kwargs['mass'].to(self.default_parameter_values['mass'].unit).value
        except:
            mass_value = kwargs['mass']


        formatted_kwargs_list = []
        for key, val in self.default_parameter_values.items():
            if hasattr(val, 'unit'):
                formatted_kwargs_list.append(kwargs[key].to(val.unit).value)
            else:
                formatted_kwargs_list.append(kwargs[key])


        for channel in self.sqrtchannelfuncdictionary.keys():

            channel_sigma = self.partial_sqrt_sigmav_interpolator_dictionary[channel]((*formatted_kwargs_list,))**2
            channel_spectrum = (self.sqrtchannelfuncdictionary[channel]((np.log10(mass_value), 
                                                           np.log10(energy)-np.log10(mass_value))))**2 # Square is to enforce positivity

            
            
            channel_comp = channel_sigma*channel_spectrum 


            # - np.log(energy) is to convert dN/dlogx to dN/dlogE = 1/(ln(E) E) dN/dlogx
            log_channel_comp = np.log(channel_comp) - np.log((energy*u.TeV).to(self.inverse_output_unit).value) - np.log(np.log(10))


            logspectra = np.logaddexp(
                logspectra, 
                log_channel_comp)
            
        
        return logspectra

    
        
    def logfunc(self, 
                energy: list | np.ndarray | float, 
                kwd_parameters: dict = {'mass':1.0},
                **kwargs) -> np.ndarray | float:
        """
        Calculates the logarithm of the gamma-ray flux.

        Args:
            energy (list | np.ndarray | float): Gamma-ray energies in TeV.
            kwd_parameters (dict, optional): Parameters for the dark matter model. Defaults to {'mass': 1.0}.

        Returns:
            np.ndarray | float: Logarithm of the gamma-ray flux.
        """

        kwd_parameters.update(kwargs)
        update_with_defaults(kwargs, self.default_parameter_values)


        
        units = [energy.unit]

        energy = np.asarray(energy)

        for key, val in kwd_parameters.items():
            if hasattr(val, 'unit'):
                unit = val.unit
            else:
                unit=1
            units.append(unit)

            
            formatted_value = np.asarray(val)

            if formatted_value.shape!=energy.shape:
                formatted_value = energy*0+formatted_value

            kwd_parameters[key] = formatted_value*unit
        

        flatten_param_vals = np.asarray([energy.flatten(), *[theta_param.flatten() for theta_param in kwd_parameters.values()]])
        
        unique_param_vals = np.unique(flatten_param_vals, axis=1)

        logspectralvals = self.spectral_gen(
            energy=unique_param_vals[0]*units[0], 
            **{param_key: unique_param_vals[1+idx].flatten()*units[1+idx] for idx, param_key in enumerate(kwd_parameters.keys())})

        mask = np.all(unique_param_vals[:, None, :] == flatten_param_vals[:, :, None], axis=0)

        slices = np.where(mask, logspectralvals[None, :], 0.0)

        logspectralvals = np.sum(slices, axis=-1).reshape(energy.shape)

        # logspectralvals = self.spectral_gen(
        #     energy=energy, 
        #     **kwd_parameters)
        
        return logspectralvals
    

    # This function presumes that it needs to create a mesh based on the input parameters
        # this is handy when one doesn't want to create a mesh that includes all the observation
        # parameter axes, spectral parameters, and spatial parameters at once, reducing dimensionality
        # and reduces the number of needed computations
    def mesh_efficient_logfunc(self, 
                               energy: list | np.ndarray | float, 
                               kwd_parameters: dict = {'mass':1.0},
                               **kwargs) -> np.ndarray | float:
        """
        Efficiently computes the log spectrum over a mesh of parameters.

        Args:
            energy (list | np.ndarray | float): Gamma-ray energies in TeV.
            kwd_parameters (dict, optional): Parameters for the dark matter model. Defaults to {'mass': 1.0}.

        Returns:
            np.ndarray | float: Logarithm of the gamma-ray flux over the parameter mesh.
        """



        kwd_parameters.update(kwargs)

        new_kwd_parameters = {param_key: np.asarray(param_val) for param_key, param_val in kwd_parameters.items()}
        units = [energy.unit]

        for param_key, param_val in kwd_parameters.items():
            if hasattr(param_val, "unit"):
                units.append(param_val.unit)
            else:
                units.append(1)

        

        param_meshes = np.meshgrid(energy.value, *new_kwd_parameters.values(), indexing='ij')

        logspectralvals = self.spectral_gen(
            energy = param_meshes[0].flatten()*units[0], 
            **{param_key: param_meshes[1+idx].flatten()*units[1+idx] for idx, param_key in enumerate(new_kwd_parameters.keys())}
            ).reshape(param_meshes[0].shape)
        
        return logspectralvals



    def mesh_integral_efficient_logfunc(self, 
                               energy: list | np.ndarray | float, 
                               kwd_parameters: dict = {'mass':1.0}) -> np.ndarray | float:
        """
        Efficiently computes the log spectrum for integration over a mesh of parameters.

        Args:
            energy (list | np.ndarray | float): Gamma-ray energies in TeV.
            kwd_parameters (dict, optional): Parameters for the dark matter model. Defaults to {'mass': 1.0}.

        Returns:
            np.ndarray | float: Logarithm of the gamma-ray flux over the parameter mesh.
        """

        energy = np.asarray(energy.value)

        new_kwd_parameters = {param_key: np.asarray(param_val) for param_key, param_val in kwd_parameters.items()}


        for key, val in new_kwd_parameters.items():
            new_kwd_parameters[key] = np.asarray(val) 

        hyper_params_shape = new_kwd_parameters[list(new_kwd_parameters.keys())[0]].shape
        # minimises the amount of needed memory and comp time by reducing number of combinations in meshgrid
        flatten_hyperparam_vals = np.asarray([*[theta_param.flatten() for theta_param in new_kwd_parameters.values()]])
            
        unique_param_vals = np.unique(flatten_hyperparam_vals, axis=1)

        param_meshes = np.meshgrid(energy, *unique_param_vals, indexing='ij')




        logspectralvals = self.spectral_gen(
            energy = param_meshes[0].flatten(), 
            **{param_key: param_meshes[1+idx].flatten() for idx, param_key in enumerate(new_kwd_parameters.keys())}
            ).reshape(param_meshes[0].shape)
        
        mask = np.all(unique_param_vals[:, None, :] == flatten_hyperparam_vals[:, :, None], axis=0)

        slices = np.where(mask, logspectralvals[:, None, :], 0.0)


        logspectralvals = np.sum(slices, axis=-1).reshape((energy.size, *hyper_params_shape))


        return logspectralvals
    

    def calc_ratios(self, kwd_parameters):
        """
        Calculates the ratios of the annihilation cross-section for each channel.

        Args:
            kwd_parameters (dict): Keyword parameters for the dark matter model.

        Returns:
            dict: Dictionary of ratios for each channel.
        """
        update_with_defaults(kwd_parameters, self.default_parameter_values)

        sigma_dict = {}

        for channel in self.sqrtchannelfuncdictionary.keys():

            channel_sigma = self.partial_sqrt_sigmav_interpolator_dictionary[channel]((*kwd_parameters.values(),))**2
            sigma_dict[channel] = channel_sigma

        return sigma_dict


