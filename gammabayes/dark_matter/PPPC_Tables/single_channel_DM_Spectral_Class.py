import os, torch
from functools import partial
from gammabayes import RegularTorchInterpolator
from gammabayes.priors.spectral_components import BaseSpectral_PriorComp
single_channel_spectral_data_path = os.path.dirname(os.path.dirname(__file__))
from .PPPC_reader import PPPCReader


class SingleDMChannel(BaseSpectral_PriorComp):
    """Class for efficient single channel dark matter spectra calculations."""

    
    def __init__(self, channel='W+W-', mass = 1.0,):
        """
        Initializes the SingleDMChannel class with specified parameters.

        Args:
            channel (str, optional): The dark matter channel to use. Defaults to 'W+W-'.
            default_parameter_values (dict, optional): Default parameter values. Defaults to {'mass': 1.0}.
        """
    
        self.channel = channel
        atprod_gammas = PPPCReader(single_channel_spectral_data_path+"/PPPC_Tables/AtProduction_gamma_EW_corrections.dat")
        atprod_mass_values = torch.tensor(atprod_gammas.mass_axis)
        atprod_log10mass_values = torch.log10(atprod_mass_values)
        atprod_log10x_values =torch.tensor(atprod_gammas.log10x_axis)

        # We take the square root of the outputs so later we can square them to enforce positivity
        try:
            PPPC_channel = PPPCReader.darkSUSY_to_PPPC_converter[self.channel]
        
            # Extracting single channel spectra
            tempspectragrid = atprod_gammas[PPPC_channel].reshape(atprod_gammas.output_shape)
            
        except:
            # Extracting single channel spectra
            tempspectragrid = atprod_gammas[self.channel].reshape(atprod_gammas.output_shape)

        tempspectragrid = torch.sqrt(torch.tensor(tempspectragrid))
            
        # Interpolating square root of PPPC tables to preserve positivity during interpolation (where result is squared)
        self.sqrtchannelfunc = interpolate.RegularTorchInterpolator(
                                    (
                                        atprod_log10mass_values, 
                                        atprod_log10x_values
                                        ), 
                                    tempspectragrid)



        # Yes ew
        self.loglog10 = torch.log(torch.log(torch.tensor(10.)))

        self.log_spectral_gen = partial(self.__log_spectral_gen, mass=mass)


    def __call__(self, *args, **kwargs):
        """
        Allows the instance to be called as a function.

        Returns:
            tensor: The result of the log function.
        """
        return self.log_spectral_gen(*args, **kwargs)



    def __log_spectral_gen(self, energy, mass=torch.tensor(1.)):
        """
        Generates the spectral values for the given energy and parameters.

        Args:
            energy (float | tensor | list): Energy values to calculate the spectrum for.

        Returns:
            tensor | float: The calculated log spectrum values.
        """
        log10mass = torch.log10(mass)

        channel_spectrum = (self.sqrtchannelfunc((log10mass, 
                                                        torch.log10(energy)-log10mass)))**2 # Square is to enforce positivity
            
        log_channel_spectrum = torch.log(channel_spectrum)

        # Converting it from dN/dlog10x to dN/dE
        log_channel_spectrum =log_channel_spectrum - torch.log(energy) - self.loglog10

        return log_channel_spectrum

    
        











