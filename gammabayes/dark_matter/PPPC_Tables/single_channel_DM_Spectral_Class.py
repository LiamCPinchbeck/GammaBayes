import os
from gammabayes import RegularTorchInterpolation
from gammabayes.priors.spectral_components import BaseSpectral
single_channel_spectral_data_path = os.path.dirname(os.path.dirname(__file__))

from .PPPCReader import PPPCReader


class SingleDMChannel(BaseSpectral_PriorComp):
    """Class for efficient single channel dark matter spectra calculations."""

    
    def __init__(self, channel='W+W-',
                 default_parameter_values = {'mass':1.0,},
                 ):
        """
        Initializes the SingleDMChannel class with specified parameters.

        Args:
            channel (str, optional): The dark matter channel to use. Defaults to 'W+W-'.
            default_parameter_values (dict, optional): Default parameter values. Defaults to {'mass': 1.0}.
        """
    
        self.channel = channel
        atprod_gammas = PPPCReader(single_channel_spectral_data_path+"/PPPC_Tables/AtProduction_gamma_EW_corrections.dat")
        atprod_mass_values = atprod_gammas.mass_axis
        atprod_log10x_values = atprod_gammas.log10x_axis

        # We take the square root of the outputs so later we can square them to enforce positivity
        try:
            PPPC_channel = PPPCReader.darkSUSY_to_PPPC_converter[self.channel]
        
            # Extracting single channel spectra
            tempspectragrid = atprod_gammas[PPPC_channel].reshape(atprod_gammas.output_shape)
            
        except:
            # Extracting single channel spectra
            tempspectragrid = atprod_gammas[self.channel].reshape(atprod_gammas.output_shape)
            
        # Interpolating square root of PPPC tables to preserve positivity during interpolation (where result is squared)
        self.sqrtchannelfunc = interpolate.RegularGridInterpolator(
            (np.log10(atprod_mass_values), atprod_log10x_values), 
            np.sqrt(np.asarray(tempspectragrid)),
            method='cubic', bounds_error=False, fill_value=0)



        # Yes ew
        self.loglog10 = torch.log(torch.log(torch.tensor(10.)))


    def __call__(self, *args, **kwargs) -> np.ndarray | float:
        """
        Allows the instance to be called as a function.

        Returns:
            np.ndarray | float: The result of the log function.
        """
        return self.logfunc(*args, **kwargs)



    def spectral_gen(self, energy, mass) -> np.ndarray | float:
        """
        Generates the spectral values for the given energy and parameters.

        Args:
            energy (float | np.ndarray | list): Energy values to calculate the spectrum for.

        Returns:
            np.ndarray | float: The calculated log spectrum values.
        """
        log10mass = np.log10(mass)

        channel_spectrum = (self.sqrtchannelfunc((log10mass, 
                                                        np.log10(energy)-log10mass)))**2 # Square is to enforce positivity
            
        log_channel_spectrum = np.log(channel_spectrum)

        # Converting it from dN/dlog10x to dN/dE
        log_channel_spectrum =log_channel_spectrum - np.log(energy) - self.loglog10

        return log_channel_spectrum

    
        
    def logfunc(self, 
                energy: list | np.ndarray | float, 
                kwd_parameters: dict = {'mass':1.0}) -> np.ndarray | float:
        """
        Calculates the log spectrum values for given energy and parameters.

        Args:
            energy (list | np.ndarray | float): Energy values to calculate the spectrum for.
            kwd_parameters (dict, optional): Keyword parameters for the calculation. Defaults to {'mass': 1.0}.

        Returns:
            np.ndarray | float: The calculated log spectrum values.
        """

        energy = np.asarray(energy.to("TeV").value)


        for key, val in kwd_parameters.items():
            kwd_parameters[key] = np.asarray(val) 


        flatten_param_vals = np.asarray([energy.flatten(), *[theta_param.flatten() for theta_param in kwd_parameters.values()]])
            

        unique_param_vals = np.unique(flatten_param_vals, axis=1)

        logspectralvals = self.spectral_gen(
            energy=unique_param_vals[0], 
            **{param_key: unique_param_vals[1+idx].flatten() for idx, param_key in enumerate(kwd_parameters.keys())})

        mask = np.all(unique_param_vals[:, None, :] == flatten_param_vals[:, :, None], axis=0)


        slices = np.where(mask, logspectralvals[None, :], 0.0)


        logspectralvals = np.sum(slices, axis=-1).reshape(energy.shape)
        
        return logspectralvals
    










