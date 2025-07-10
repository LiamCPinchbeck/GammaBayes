import numpy as np, time
from gammabayes.dark_matter.spectral_models import (
    DM_ContinuousEmission_Spectrum
)
from gammabayes.likelihoods.irfs import IRF_LogLikelihood
from os import path
from gammabayes.dark_matter.density_profiles import DM_Profile
from gammabayes.priors.core import TwoCompFluxPrior
from gammabayes import update_with_defaults
from gammabayes.utils import logspace_simpson
from astropy import units as u

class CombineDMComps(TwoCompFluxPrior):
    """
    A class to combine dark matter spectral and spatial components for analysis.

    Inherits from:
        TwoCompPrior

    Methods:
        __init__(self, spectral_class: DM_ContinuousEmission_Spectrum, spatial_class: DM_Profile, *args, **kwargs)
        convert_param_to_logsigmav(self, signal_fraction, true_axes=None, spectral_parameters={}, spatial_parameters={}, totalnumevents=1e8, tobs_seconds=525*60*60, symmetryfactor=1)
        calc_ratios(self, kwd_parameters={})
    """

    def __init__(self, 
                 spectral_class: DM_ContinuousEmission_Spectrum, 
                 spatial_class: DM_Profile, 
                 irf_loglike: IRF_LogLikelihood,
                 *args, **kwargs
                 ):
        """
        Initializes the CombineDMComps object with the given spectral and spatial classes.

        Args:
            spectral_class (DM_ContinuousEmission_Spectrum): The spectral component class.
            spatial_class (DM_Profile): The spatial component class.
            *args: Additional positional arguments to pass to the parent class.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        

        super().__init__(spectral_class=spectral_class, 
                         spatial_class =spatial_class,
                         irf_loglike=irf_loglike,
                         *args, **kwargs
        )

    def convert_param_to_sigmav(self,
                            signal_fraction, 
                            true_axes=None,
                            spectral_parameters = {},
                            spatial_parameters = {},
                            totalnumevents=1e8, 
                            tobs_seconds=525*60*60, symmetryfactor=1):
        """
        Converts signal fraction parameters to the anihilation/decay cross-section.

        Args:
            signal_fraction (float): The fraction of the signal.
            true_axes (list, optional): The true axes values. Defaults to None.
            spectral_parameters (dict, optional): Spectral parameters. Defaults to {}.
            spatial_parameters (dict, optional): Spatial parameters. Defaults to {}.
            totalnumevents (float, optional): The total number of events. Defaults to 1e8.
            tobs_seconds (float, optional): The observation time in seconds. Defaults to 525*60*60.
            symmetryfactor (int, optional): The symmetry factor. Defaults to 1.

        Returns:
            np.ndarray: The cross-section values.
        """
        
        update_with_defaults(spectral_parameters, self.default_spectral_parameters)
        update_with_defaults(spatial_parameters, self.default_spatial_parameters)


        if true_axes is None:
            true_axes = self.binning_geometry.axes
        

        if self.mesh_efficient_exists:
            integrand = self.log_dist_mesh_efficient(*true_axes, 
                                                     spectral_parameters=spectral_parameters,
                                                     spatial_parameters=spatial_parameters)
            
        else:
            true_axis_mesh = np.meshgrid(*true_axes, indexing='ij')
            true_axis_mesh_flattened = [mesh.flatten() for mesh in true_axis_mesh]

            integrand = self.log_dist(*true_axis_mesh_flattened, 
                                                     spectral_parameters=spectral_parameters,
                                                     spatial_parameters=spatial_parameters).reshape(true_axis_mesh[0].shape)

        # Splitting it up for easy debuggin
        log_integrated_energy = logspace_simpson(
                                    logy=integrand, x = true_axes[0].value, axis=0)
                
        log_integrated_energy_longitude = logspace_simpson(
                                logy=log_integrated_energy, 
                                    x = true_axes[1].value, axis=0)
                                
        logintegral = logspace_simpson(
                            logy=log_integrated_energy_longitude.T*np.cos(true_axes[2].to(u.rad)), 
                                    x = true_axes[2].value, axis=-1).T 
        

        logsigmav = np.log(8*np.pi*symmetryfactor*spectral_parameters['mass'].value**2*totalnumevents*signal_fraction) - logintegral - np.log(tobs_seconds)


        return np.exp(logsigmav)


    
    def calc_ratios(self, kwd_parameters={}):
        """
        Calculates the ratios for the spectral component based on given parameters.

        Args:
            kwd_parameters (dict, optional): Keyword parameters for the spectral component. Defaults to {}.

        Returns:
            dict: The calculated ratios for the spectral component.
        """
        update_with_defaults(kwd_parameters, self.default_spectral_parameters)

        return self.spectral_comp.calc_ratios(kwd_parameters)
    

