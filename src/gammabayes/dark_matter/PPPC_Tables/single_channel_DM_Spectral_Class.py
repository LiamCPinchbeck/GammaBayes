import os, torch
from functools import partial
from gammabayes import RegularTorchInterpolator
from gammabayes.priors.spectral_components import BaseSpectral_PriorComp
single_channel_spectral_data_path = os.path.dirname(os.path.dirname(__file__))
from .PPPC_reader import PPPCReader


class SingleDMSpectralComp(BaseSpectral_PriorComp):
    """Class for efficient single channel dark matter spectra calculations."""

    
    def __init__(self, binning_geometry, channel='W+W-', mass = torch.tensor(1.), *args, **kwargs):
        """
        Initializes the SingleDMChannel class with specified parameters.

        Args:
            channel (str, optional): The dark matter channel to use. Defaults to 'W+W-'.
            default_parameter_values (dict, optional): Default parameter values. Defaults to {'mass': 1.0}.
        """
    
        self.channel = channel
        atprod_gammas = PPPCReader(single_channel_spectral_data_path+"/PPPC_Tables/AtProduction_gamma_EW_corrections.dat")
        self._atprod_mass_values = torch.tensor(atprod_gammas.mass_axis)
        self._atprod_log10mass_values = torch.log10(self._atprod_mass_values)
        self._atprod_log10x_values =torch.tensor(atprod_gammas.log10x_axis)

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
        self.sqrtchannelfunc = RegularTorchInterpolator(
                                    (
                                        self._atprod_log10mass_values, 
                                        self._atprod_log10x_values
                                        ), 
                                    tempspectragrid)



        # Yes ew
        self.loglog10 = torch.log(torch.log(torch.tensor(10.)))

        mass = torch.ones_like(binning_geometry.energy_axis)

        self.log_spectral_gen = partial(self.__log_spectral_gen, mass=mass)

        super().__init__(
            logfunc = self.log_spectral_gen,

        )


    def __log_spectral_gen(self, energy, mass):

        log10mass = torch.log10(mass)

        channel_spectrum = (self.sqrtchannelfunc((log10mass, 
                                                        torch.log10(energy)-log10mass)))**2 # Square is to enforce positivity
            
        log_channel_spectrum = torch.log(channel_spectrum)

        # Converting it from dN/dlog10x to dN/dE
        log_channel_spectrum =log_channel_spectrum - torch.log(energy) - self.loglog10

        log_channel_spectrum = torch.where(energy>mass, -50, log_channel_spectrum)

        return log_channel_spectrum

    
        











