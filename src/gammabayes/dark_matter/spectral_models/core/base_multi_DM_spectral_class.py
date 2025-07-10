from gammabayes.dark_matter.channel_spectra import (
    single_channel_spectral_data_path,
    PPPCReader, 
)
from scipy import interpolate
import numpy as np

class multi_comp_dm_spectrum(object):
    """
    Class to handle multi-component dark matter spectra calculations.

    Attributes:
        micrOMEGAs_to_darkSUSY (dict): Dictionary to convert micrOMEGAs channels to darkSUSY channels.
    """


    micrOMEGAs_to_darkSUSY = {
        'AA':'gammagamma',
        'AW+W-':['gamma', 'W+W-'],
        'GG':'gg',
        'W+W-':'W+W-',
        'ZZ':'ZZ',
        'bB':'bb',
        'cC':'cc',
        'dD':'dd',
        'hX1':['h'],
        'hh':'HH',
        'hx1':['h'],
        'hx2':['h'],
        'lL':'tau+tau-',
        'mM':"mu+mu-",
        'sS':'ss',
        'tT':'tt',
        'uU':'uu',
    }


    def __init__(self, annihilation_ratios_nested_dict:dict, parameter_interpolation_values:list):
        """
        Initializes the multi-component dark matter spectrum class.

        Args:
            annihilation_ratios_nested_dict (dict): Nested dictionary of annihilation ratios.
            parameter_interpolation_values (list): List of parameter interpolation values.
        """

        # Extracting unique values
        atprod_gammas = PPPCReader(single_channel_spectral_data_path+"/PPPC_Tables/AtProduction_gamma_EW_corrections.dat")
        atprod_mass_values = atprod_gammas.mass_axis
        atprod_log10x_values = atprod_gammas.log10x_axis


        # We take the square root of the outputs so later we can square them to enforce positivity
        sqrtchannelfuncdictionary = {}
        for micrOMEGAs_channel, darkSUSYchannel in self.micrOMEGAs_to_darkSUSY.items():
            try:
                if type(darkSUSYchannel)==str:
                    PPPC_channel = PPPCReader.darkSUSY_to_PPPC_converter[darkSUSYchannel]
                    
                    # Extracting single channel spectra
                    sqrt_tempspectragrid = np.sqrt(atprod_gammas[PPPC_channel].reshape(atprod_gammas.output_shape)/1000) # 1000 is to convert to 1/TeV 
                    
                    # Interpolating square root of PPPC tables to preserve positivity during interpolation (where result is squared)
                    sqrtchannelfuncdictionary[micrOMEGAs_channel] = interpolate.RegularGridInterpolator(
                        (np.log10(atprod_mass_values), atprod_log10x_values), 
                        sqrt_tempspectragrid,
                        method='cubic', bounds_error=False, fill_value=0)
                else:
                    darkSUSYchannels = darkSUSYchannel
                    PPPC_channels = [PPPCReader.darkSUSY_to_PPPC_converter[darkSUSYchannel] for darkSUSYchannel in darkSUSYchannels]
                    
                    # Extracting single channel spectra
                    # 1000 is to convert to 1/TeV 
                    tempspectragrids = [np.sqrt(atprod_gammas[PPPC_channel].reshape(atprod_gammas.output_shape)/1000) for PPPC_channel in PPPC_channels] 
                    
                    tempspectragrid = np.sum(tempspectragrids, axis=0)

                    if len(darkSUSYchannels)==1:
                        tempspectragrid = 0.5*tempspectragrid
                    
                    sqrt_tempspectragrid = np.sqrt(tempspectragrid)

                    # Interpolating square root of PPPC tables to preserve positivity during interpolation (where result is squared)
                    sqrtchannelfuncdictionary[micrOMEGAs_channel] = interpolate.RegularGridInterpolator(
                        (np.log10(atprod_mass_values), atprod_log10x_values), 
                        sqrt_tempspectragrid,
                        method='cubic', bounds_error=False, fill_value=0)            
            
            except:
                sqrtchannelfuncdictionary[micrOMEGAs_channel] = self.zero_output

        self.sqrtchannelfuncdictionary = sqrtchannelfuncdictionary
        # sqrt enforces positivity while also transforming values to closer to 1
        self.partial_sqrt_sigmav_interpolator_dictionary = {
            initial_state_key: {
            final_MM_state_key: 
            interpolate.RegularGridInterpolator(
                (*parameter_interpolation_values,),
                np.sqrt(annihilation_ratios_nested_dict[initial_state_key][final_MM_state_key]),
                method='linear', bounds_error=False, fill_value=0) \
                    for final_MM_state_key in self.micrOMEGAs_to_darkSUSY.keys()
                } for initial_state_key in annihilation_ratios_nested_dict.keys()}
        
        self.annihilation_ratios_nested_dict = annihilation_ratios_nested_dict


    def __call__(self, *args, **kwargs) -> np.ndarray | float:
        """
        Allows the instance to be called as a function, delegating to the `logfunc` method.

        Args and Returns:
            See `logfunc` method for detailed argument and return value descriptions.
        """
        return self.logfunc(*args, **kwargs)


    
    def zero_output(self, inputval):
        """
        Returns zero for given input.

        Args:
            inputval: Input value.

        Returns:
            Zero value.
        """
        return inputval[0]*0

    def one_output(self, inputval):
        """
        Returns one for given input.

        Args:
            inputval: Input value.

        Returns:
            One value.
        """
        return inputval[0]*0 + 1
    

    def logfunc(self, energy, mDM1, mDM2):
        """
        Computes the log spectrum for the given energy and dark matter masses.

        Args:
            energy: Energy values.
            mDM1: Mass of the first dark matter particle.
            mDM2: Mass of the second dark matter particle.

        Returns:
            Log spectrum values.
        """

        log_spectrum = -np.inf


            
        for initial_state_key  in self.annihilation_ratios_nested_dict.keys():

            if initial_state_key.lower()=='x1x1':
                com_energy = mDM1 #mdm1, mdm2
            elif initial_state_key=='x1X2':
                com_energy = np.sqrt(mDM1*mDM2) #mdm1, mdm2
            elif initial_state_key=='x2X2':
                com_energy = mDM2  #mdm1, mdm2
            else:
                print(f'unknown initial_state: {initial_state_key}')

            sqrt_ratio_funcs_for_initial_state = self.partial_sqrt_sigmav_interpolator_dictionary[initial_state_key]

            for final_state_key in self.micrOMEGAs_to_darkSUSY.keys():
                log_bf = np.log(sqrt_ratio_funcs_for_initial_state[final_state_key]((mDM1, mDM2))**2)
                
                sqrt_single_chan_spec_func = self.sqrtchannelfuncdictionary[final_state_key]

                log_single_chan_spec = np.log(sqrt_single_chan_spec_func((np.log10(com_energy), np.log10(energy)-np.log10(com_energy)))**2)
                    
                log_spectrum = np.logaddexp(log_spectrum, log_bf + log_single_chan_spec)
                

        return log_spectrum
    
    def mesh_efficient_logfunc(self, 
                               energy: list | np.ndarray | float, 
                               kwd_parameters: dict = {'mDM1':5.0, 'mDM2': 7.0}) -> np.ndarray | float:
        """
        Efficiently computes the log spectrum over a mesh of parameters.

        Args:
            energy (list | np.ndarray | float): Energy values.
            kwd_parameters (dict, optional): Dictionary of keyword parameters, defaults to {'mDM1': 5.0, 'mDM2': 7.0}.

        Returns:
            np.ndarray | float: Log spectrum values.
        """

        kwd_parameters = {param_key: np.asarray(param_val) for param_key, param_val in kwd_parameters.items()}

        param_meshes = np.meshgrid(energy, *kwd_parameters.values(), indexing='ij')

        logspectralvals = self(
            energy = param_meshes[0].flatten(), 
            **{param_key: param_meshes[1+idx].flatten() for idx, param_key in enumerate(kwd_parameters.keys())}
            ).reshape(param_meshes[0].shape)
        
        return logspectralvals
