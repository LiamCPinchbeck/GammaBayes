import numpy as np
import time, random, warnings, os
from scipy import special
from astropy import units as u
from gammabayes.dark_matter.channel_spectra import (
    single_channel_spectral_data_path,
    PPPCReader, 
)
from gammabayes.dark_matter.density_profiles import Einasto_Profile
from gammabayes.priors import DiscreteLogPrior, SourceFluxDiscreteLogPrior
from gammabayes.priors.core.observation_flux_prior import ObsFluxDiscreteLogPrior

# $$
from gammabayes.likelihoods.irfs import IRF_LogLikelihood

from gammabayes.utils import (
    dynamic_import,
)
from gammabayes.hyper_inference import MTree

from gammabayes import (
    GammaBinning, GammaLogExposure, GammaObs, GammaObsCube,
    Parameter, ParameterSet, ParameterSetCollection,
)
from matplotlib import pyplot as plt
from gammabayes.dark_matter import CustomDMRatiosModel, CombineDMComps

from gammabayes.priors.astro_sources import construct_fermi_gaggero_flux_matrix, construct_hess_flux_matrix


from scipy.interpolate import RegularGridInterpolator
from gammabayes.likelihoods.irfs import FOV_IRF_Norm
from gammabayes.hyper_inference import DiscreteAdaptiveScan, ScanOutput_StochasticTreeMixturePosterior

from .config_utils import from_config_dict, from_config_file

from icecream import ic

class log_obs_interpolator:
    """
    Class for creating a log-space interpolator for observational data.

    Args:
        real_space_interpolator (callable): A function for real-space interpolation.
    """

    def __init__(self, real_space_interpolator):
        """_summary_

        Args:
            real_space_interpolator (_type_): _description_
        """
        self.real_space_interpolator = real_space_interpolator

    def log_func(self, energy, lon, lat, spectral_parameters={}, spatial_parameters={}):
        """
        Logarithmic interpolation function.

        Args:
            energy (_type_): Energy values for interpolation.
            lon (_type_): Longitude values for interpolation.
            lat (_type_): Latitude values for interpolation.
            spectral_parameters (dict, optional): Spectral parameters. Defaults to {}.
            spatial_parameters (dict, optional): Spatial parameters. Defaults to {}.

        Returns:
            _type_: Interpolated log-space values.
        """

        output = np.log(self.real_space_interpolator( (energy, lon, lat) ))

        return output



class High_Level_Setup:


    def __init__(self,
                 true_binning_geometry:GammaBinning, 
                 recon_binning_geometry:GammaBinning,
                 prior_parameter_specifications: dict|ParameterSetCollection,
                 shared_parameter_specifications: dict,
                 mixture_parameter_specifications: dict|ParameterSet,
                 dark_matter_model_specifications: dict,
                 pointing_dirs: list[np.ndarray[u.Quantity]],
                 observation_times: list[u.Quantity]|np.ndarray[u.Quantity],
                 observational_prior_models: list[str, DiscreteLogPrior],
                 log_exposures: list[GammaLogExposure]=None,
                 observation_meta: dict={},
                 observation_cube:GammaObsCube|list[GammaObs]=None,
                 mixture_layout:dict = None,
                 marginalisation_method = DiscreteAdaptiveScan,
                 marginalisation_bounds = [['log10', 1.0*u.TeV], ['linear', 1.0*u.deg], ['linear', 1.0*u.deg]],
                 save_path:str='',
                 skip_irf_norm_setup:bool=False,
                 true_mixture_specifications:dict = {},
                 irf_log_norm_matrix=None,
                 edisp_log_norm_matrix=None,
                 psf_log_norm_matrix=None,
                 config_dict:dict = {},
                 **kwargs
                 ):
        
        np.seterr(divide='ignore')
        
        self.config_dict = config_dict
        self.config_dict.update(kwargs)
        self.observation_meta = observation_meta
        

        self.true_binning_geometry = true_binning_geometry
        self.recon_binning_geometry = recon_binning_geometry


        self.observation_cube = self.setup_observation_cube(pointing_dirs=pointing_dirs, 
                                                            observation_times=observation_times, 
                                                            log_exposures=log_exposures, 
                                                            observation_meta=observation_meta, 
                                                            observation_cube=observation_cube,)


        self.observation_cube.binning_geometry = self.recon_binning_geometry
        self.observation_cube.meta['true_binning_geometry'] = self.true_binning_geometry

        self.dark_matter_model_specifications = dark_matter_model_specifications

        self.shared_parameter_specifications = shared_parameter_specifications
        self.mixture_parameter_specifications = ParameterSet(mixture_parameter_specifications)

        self._obs_prior_parameter_specifications = prior_parameter_specifications


        for obs_index, observation in enumerate(self.observation_cube):
            self.observation_cube[obs_index].meta['priors'] = self.setup_priors_for_observation(
                base_priors=observational_prior_models,
                observation=observation)
            


        self._setup_parameter_specifications()


        self.mixture_tree = self._setup_mixture_tree(mixture_layout, true_mixture_specifications)


        self.save_path = save_path

        self._psf_log_norm_matrix = psf_log_norm_matrix
        self._edisp_log_norm_matrix = edisp_log_norm_matrix
        self._irf_log_norm_matrix = irf_log_norm_matrix


        if not(skip_irf_norm_setup):


            temp_irf_loglike = self.observation_cube[0].irf_loglike

            self._setup_irf_norm(irf_loglike=temp_irf_loglike)


        self.marginalisation_method = marginalisation_method
        if isinstance(self.marginalisation_method, str):
            self.marginalisation_method = dynamic_import("gammabayes.hyper_inference", self.marginalisation_method)

        self.marginalisation_bounds = marginalisation_bounds

        for _axis_idx, axis_bound_info in enumerate(self.marginalisation_bounds):
            self.marginalisation_bounds[_axis_idx][1] = u.Quantity(axis_bound_info[1])

        

    from_config_file = classmethod(from_config_file)
    from_config_dict = classmethod(from_config_dict)





    def setup_observation_cube(self, pointing_dirs, observation_times, 
                               log_exposures, observation_meta, observation_cube):
        
        if observation_cube is None:

            if log_exposures is None:
                log_exposures = [None]*len(pointing_dirs)
            
            observations = []

            for pointing_dir, observation_time, log_exposure in zip(pointing_dirs, observation_times, log_exposures):
                
                irf_loglike = IRF_LogLikelihood(
                    axes=self.recon_binning_geometry.axes,
                    dependent_axes=self.true_binning_geometry.axes,
                    pointing_dir=pointing_dir,
                      observation_time=observation_time,
                      **observation_meta)
                
                if log_exposure is None:
                    log_exposure = GammaLogExposure(binning_geometry=self.true_binning_geometry,
                                                    irfs=irf_loglike,
                                                    pointing_dir=pointing_dir,
                                                    observation_time=observation_time,)


                new_obs = GammaObs(binning_geometry=self.recon_binning_geometry,
                                    pointing_dir=pointing_dir,
                                    observation_time=observation_time,
                                    irf_loglike=irf_loglike,
                                    log_exposure=log_exposure,
                                    meta={
                                        'true_binning_geometry':self.true_binning_geometry
                                    })
                                
                observations.append(new_obs)

            observation_cube = GammaObsCube(observations=observations,
                                            binning_geometry=self.recon_binning_geometry,
                                            pointing_dirs=pointing_dirs,
                                            observation_times=observation_times)
        elif isinstance(observation_cube, list):

            observation_cube = GammaObsCube(observations=observation_cube,
                                            binning_geometry=self.recon_binning_geometry,
                                            pointing_dirs=pointing_dirs,
                                            observation_times=observation_times)

        return observation_cube


    def setup_prior_from_str(self, model_name:str, irf_loglike:IRF_LogLikelihood, observation:GammaObs):

                
        if not(model_name in ['DM', 'CCR_BKG', 'FG_HS_CCR', 'FG_HS']):

            prior_model_class = dynamic_import('gammabayes.priors', model_name)
            prior_model = prior_model_class(
                axes=self.true_binning_geometry.axes,
                binning_geometry=self.true_binning_geometry,
                irf_loglike=irf_loglike,
                pointing_dir=observation.pointing_dir,
                observation_time=observation.observation_time,
                log_exposure_map=observation.log_exposure)
            
            return {model_name:prior_model}

    
        elif model_name=='DM':

            self._handle_dm_specifications()

            if not(self.use_custom_dm_channels):
                prior_model = CombineDMComps(
                    name=f'DM',

                    spectral_class = self.dark_matter_spectral_class,
                    spatial_class = self.dark_matter_density_profile,

                    axes=self.true_binning_geometry.axes,

                    irf_loglike = irf_loglike, 
                    pointing_dir=observation.pointing_dir,
                    observation_time=observation.observation_time,
                    log_exposure_map=observation.log_exposure,

                    default_spectral_parameters={'mass':self.dark_matter_mass}
                    )
                
                return {prior_model.name:prior_model}
            else:

                custom_ratios_model = CustomDMRatiosModel(
                    name='DM',
                    channels=self.DM_Channels,
                    irf_loglike=irf_loglike, 
                    spatial_class=self.dark_matter_density_profile,
                    axes=self.true_binning_geometry.axes, 
                    pointing_dir=observation.pointing_dir,
                    observation_time=observation.observation_time,
                    default_spectral_parameters={'mass':self.dark_matter_mass},
                    log_exposure_map=observation.log_exposure)

                return custom_ratios_model.channel_prior_dict


        elif model_name=='CCR_BKG':

            ccr_bkg_prior = ObsFluxDiscreteLogPrior(
                logfunction=irf_loglike.log_bkg_CCR, 
                name='CCR_BKG',
                axes=self.true_binning_geometry.axes, 
                pointing_dir=observation.pointing_dir,
                observation_time=observation.observation_time)
            
            return {ccr_bkg_prior.name:ccr_bkg_prior}


        elif 'FG' in model_name and 'HS' in model_name:

            true_meshes = np.meshgrid(*self.true_binning_geometry.axes, indexing='ij')

            log_fermi_gaggero_flux_matrix = np.log(construct_fermi_gaggero_flux_matrix(binning_geometry = self.true_binning_geometry))
            log_hess_source_flux_matrix = np.log(construct_hess_flux_matrix(binning_geometry = self.true_binning_geometry))
            
            log_fermi_gaggero_rate_matrix = log_fermi_gaggero_flux_matrix+observation.log_exposure
            log_hess_source_rate_matrix = log_hess_source_flux_matrix+observation.log_exposure

            log_bkg_matrix = special.logsumexp([np.log(log_fermi_gaggero_rate_matrix.value), 
                                                np.log(log_hess_source_rate_matrix.value)], axis=0)


            if 'CCR' in model_name:
                log_CCR_matrix = irf_loglike.log_bkg_CCR(energy=true_meshes[0], longitude=true_meshes[1], latitude=true_meshes[2])


                log_bkg_matrix = special.logsumexp([log_bkg_matrix, log_CCR_matrix], axis=0)

            # Pseudo-normalisation to regularise/make calculations more numerically stable
            log_bkg_matrix = log_bkg_matrix - special.logsumexp(log_bkg_matrix)

            fixed_background_interpolator = RegularGridInterpolator(points=self.true_binning_geometry.axes, 
                                                                    values=np.exp(log_bkg_matrix))


            log_obs_interpolator_bkgs = log_obs_interpolator(fixed_background_interpolator)



            bkg_prior = DiscreteLogPrior(
                logfunction=log_obs_interpolator_bkgs.log_func, 
                name=model_name,
                axes=self.true_binning_geometry.axes)
            
            return {bkg_prior.name:bkg_prior}




    def setup_priors_for_observation(self, 
                                     base_priors:list,
                                     observation:GammaObs):
        """
        Sets up observational prior models.
        """

        irf_loglike = observation.irf_loglike

        irf_loglike.pointing_dir = observation.pointing_dir


        observation_level_prior_models = {}

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for model in base_priors:
            if isinstance(model, str):
                observation_level_prior_models.update(
                    self.setup_prior_from_str(
                        model_name=model,
                        irf_loglike=irf_loglike,
                        observation=observation,))


            elif isinstance(model, DiscreteLogPrior):
                observation_level_prior_models[model.name] = model

        warnings.filterwarnings("default", category=RuntimeWarning)

        return observation_level_prior_models



    def _setup_parameter_specifications(self):
        """
        Sets up parameter specifications for the priors.
        """

        observational_prior_models = self.observation_cube.observations[0].meta['priors']
        num_obs_priors = len(observational_prior_models)

        formatted_prior_parameter_specifications = {}


        for model_idx, obs_prior in enumerate(observational_prior_models.values()):

            if obs_prior.name in self._obs_prior_parameter_specifications:
                unique_parameter_spec = ParameterSet(self._obs_prior_parameter_specifications[obs_prior.name])
            else:
                unique_parameter_spec = ParameterSet()

            formatted_prior_parameter_specifications[obs_prior.name] = unique_parameter_spec


        for parameter_name, [shared_models, parameter_spec] in self.shared_parameter_specifications.items():
            if not('name' in parameter_spec):
                parameter_spec['name'] = parameter_name


            for shared_model_name in shared_models:
                
                parameter_spec['prior_id'] = shared_model_name


                formatted_prior_parameter_specifications[shared_model_name].append(Parameter(parameter_spec))


        self.obs_prior_parameter_specifications = list(formatted_prior_parameter_specifications.values())

    


    def _handle_dm_specifications(self):
        """
        Handles dark matter specifications.
        """

        if 'dark_matter_density_profile' in self.dark_matter_model_specifications:
            density_profile_string = self.dark_matter_model_specifications['dark_matter_density_profile']
            self.dark_matter_density_profile = dynamic_import(
                'gammabayes.dark_matter.density_profiles',  density_profile_string)
        else:
            density_profile_string = 'Einasto_Profile'
            self.dark_matter_density_profile = Einasto_Profile



        if 'dark_matter_mass' in self.dark_matter_model_specifications:
            self.dark_matter_mass = u.Quantity(self.dark_matter_model_specifications['dark_matter_mass'])

        else:
            self.dark_matter_mass = np.nan



        if 'dark_matter_spectral_model' in self.dark_matter_model_specifications:
            self.use_custom_dm_channels = False
            dm_spectral_model_string = self.dark_matter_model_specifications['dark_matter_spectral_model']

            self.dark_matter_spectral_class = dynamic_import(
                'gammabayes.dark_matter.spectral_models',  dm_spectral_model_string)
            

            # self.dark_matter_model = CombineDMComps(
            #     name=f'DM',

            #     spectral_class = self.dark_matter_spectral_class,
            #     spatial_class = self.dark_matter_density_profile,

            #     axes=self.true_binning_geometry.axes,

            #     irf_loglike = self.irf_loglike, 

            #     default_spectral_parameters={'mass':self.dark_matter_mass}
            #     )
            

        elif 'DM_Channels' in self.dark_matter_model_specifications:
            self.use_custom_dm_channels = True

            self.DM_Channels = self.dark_matter_model_specifications['DM_Channels']

            if 'DM_Annihilation_Ratios' in self.dark_matter_model_specifications:
                self.DM_Annihilation_Ratios = self.dark_matter_model_specifications['DM_Annihilation_Ratios']
            else:
                self.DM_Annihilation_Ratios = None

            # self.custom_ratios_model = CustomDMRatiosModel(
            #     name='DM',
            #     channels=self.DM_Channels,
            #     irf_loglike=self.irf_loglike, 
            #     spatial_class=self.dark_matter_density_profile,
            #     axes=self.true_binning_geometry.axes, 
            #     default_spectral_parameters={'mass':self.dark_matter_mass})

        if self.use_custom_dm_channels:
            if not('DM' in self.mixture_parameter_specifications):
                warnings.warn(
                    """When using dark matter models by channel, usually there is a specification for the
                    overall signal fraction named 'DM' within the mixture parameter specifications.""", 
                    category=UserWarning)
                


    
    def _setup_mixture_tree(self, layout, specifications):
        """
        Sets up the mixture tree values.

        Args:
            values (iterable): Values to set up in the mixture tree. Defaults to None.
        """



        unformatted_values = []

        true_mixture_fractions = specifications.get('true_mixture_fractions', {})

        DM_Annihilation_Ratios = specifications.get('DM_Annihilation_Ratios', {})


        values = []
        for source_name, fraction in true_mixture_fractions.items():


            values.append(fraction)

            if source_name=='DM':
                values.extend(list(DM_Annihilation_Ratios.values()))

        if len(values) <1:
            values=None


        mixture_tree = MTree()
        mixture_tree.create_tree(layout, values=values)

        print("__________ MIXTURE TREE STRUCTURE __________")
        print("""note: if 'True' values have been unspecified, 
then the tree node values are the defaults such that if you add them up for their given branch you get 1.\n""")
        

        return mixture_tree

                

    def _setup_irf_norm(self, irf_loglike,  
                        irf_log_norm_matrix=None, edisp_log_norm_matrix=None, psf_log_norm_matrix=None, 
                        base_pointing_dir=None, pointing_dirs=None):
        


        if pointing_dirs is None:
            pointing_dirs = self.observation_cube.pointing_dirs

        if base_pointing_dir is None:
            base_pointing_dir = self.observation_cube.central_pointing

        if irf_log_norm_matrix is None:
            irf_log_norm_matrix = self._irf_log_norm_matrix

        if edisp_log_norm_matrix is None:
            edisp_log_norm_matrix = self._edisp_log_norm_matrix

        if psf_log_norm_matrix is None:
            psf_log_norm_matrix = self._psf_log_norm_matrix


        irf_loglike.pointing_dir = base_pointing_dir


        self.fov_irf_norm = FOV_IRF_Norm(true_binning_geometry=self.true_binning_geometry, 
                recon_binning_geometry=self.recon_binning_geometry,
                original_norm_matrix_pointing_dir = base_pointing_dir,
                new_pointing = base_pointing_dir,
                pointing_dirs = pointing_dirs,
                irf_loglike=irf_loglike,
                log_edisp_norm_matrix = edisp_log_norm_matrix,
                log_psf_norm_matrix = psf_log_norm_matrix,
                irf_norm_matrix=irf_log_norm_matrix
                )
        
    @property
    def irf_log_norm(self):

        if not(hasattr(self, 'fov_irf_norm')):
            self._setup_irf_norm(self, self.observation_cube.observations[0].irf_loglike)

        return self.fov_irf_norm
    


    