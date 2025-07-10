import numpy as np
import time, random, warnings, os
from scipy import special

from gammabayes.dark_matter.channel_spectra import (
    single_channel_spectral_data_path,
    PPPCReader, 
)
from gammabayes.dark_matter.density_profiles import Einasto_Profile
from gammabayes.priors import DiscreteLogPrior, SourceFluxDiscreteLogPrior
from gammabayes.priors.core.observation_flux_prior import ObsFluxDiscreteLogPrior

# $$
from gammabayes.likelihoods.irfs import IRF_LogLikelihood
from gammabayes.utils.config_utils import (
    read_config_file, 
    create_true_axes_from_config, 
    create_recon_axes_from_config, 
)
from gammabayes.utils import (
    dynamic_import,
)
from gammabayes.hyper_inference import MTree

from gammabayes import (
    GammaBinning, GammaLogExposure, GammaObs, GammaObsCube,
    Parameter, ParameterSet, ParameterSetCollection,
    apply_dirichlet_stick_breaking_direct,
    update_with_defaults
)
from matplotlib import pyplot as plt
from gammabayes.dark_matter import CustomDMRatiosModel, CombineDMComps

from gammabayes.priors.astro_sources import construct_fermi_gaggero_flux_matrix, construct_hess_flux_matrix
from gammabayes.utils.integration import iterate_logspace_integration, logspace_simpson


from dynesty.pool import Pool as DyPool
from scipy.interpolate import RegularGridInterpolator



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


class hl_setup_from_config:
    """
    High-level setup class from configuration file.

    Args:
        config (_type_): Configuration file or dictionary.
    """

    def __init__(self, config, skip_irf_norm_setup=False):
        """_summary_

        Args:
            config (_type_): _description_
        """

        print("\n\n")
        if isinstance(config, dict):
            self.config_dict = config
        else:
            self.config_dict = read_config_file(config)

        self.prior_parameter_specifications = self.config_dict['prior_parameter_specifications']
        self.shared_parameter_specifications = self.config_dict['shared_parameter_specifications']
        self.mixture_parameter_specifications = self.config_dict['mixture_parameter_specifications']
        

        self.save_path = self.config_dict['save_path']


        self.true_axes       = create_true_axes_from_config(self.config_dict)
        self.recon_axes      = create_recon_axes_from_config(self.config_dict)

        self.true_binning = GammaBinning(*self.true_axes)
        self.recon_binning = GammaBinning(*self.recon_axes)

        self._setup_mixture_tree_values() 
        self._setup_irfs(skip_irf_norm_setup=skip_irf_norm_setup)
        self._handle_missing_true_specifications()
        self._handle_dm_specifications()
        self._setup_observational_prior_models()
        self._setup_parameter_specifications()
        



        
    def _setup_irfs(self, skip_irf_norm_setup=False):
        """
        Sets up the Instrument Response Functions (IRFs).
        """
        

        if 'IRF_kwarg_Specifications' in self.config_dict:
            IRF_kwarg_Specifications = self.config_dict['IRF_kwarg_Specifications']


        self.irf_loglike = IRF_LogLikelihood(axes   =   self.recon_axes, 
                                dependent_axes =   self.true_axes,
                                **IRF_kwarg_Specifications)
        

        if not(skip_irf_norm_setup):
            if 'log_irf_norm_matrix_path' in self.config_dict:
                self.log_irf_norm_matrix_path = self.config_dict['log_irf_norm_matrix_path']

                print(f"\n__Log IRF Matrix loaded from specified path__: {self.log_irf_norm_matrix_path}\n")

                try:
                    self.log_irf_norm_matrix = np.load(self.log_irf_norm_matrix_path)

                except:
                    self.log_irf_norm_matrix = None
            else:
                self.log_irf_norm_matrix = None


            if ('log_edisp_norm_matrix_path' in self.config_dict) and ('log_psf_norm_matrix_path' in self.config_dict):
                self.log_psf_norm_matrix_load_path = self.config_dict['log_psf_norm_matrix_path']
                self.log_psf_norm_matrix = np.load(self.log_psf_norm_matrix_load_path)

                self.log_edisp_norm_matrix_path = self.config_dict['log_edisp_norm_matrix_path']
                self.log_edisp_norm_matrix = np.load(self.log_edisp_norm_matrix_path)

                print(f"""\n__Log IRF energy dispersion and PSF matrices loaded from specified paths__:\n
    PSF: {self.log_psf_norm_matrix_load_path}
    EDISP: {self.log_edisp_norm_matrix_path}\n\n\n""")
                
            else:

                self.log_psf_norm_matrix = None
                self.log_edisp_norm_matrix = None



            if self.log_irf_norm_matrix is None:
                if self.log_psf_norm_matrix is None:
                    self.log_psf_norm_matrix, self.log_edisp_norm_matrix = self.irf_loglike.create_log_norm_matrices()

                self.log_irf_norm_matrix = self.log_psf_norm_matrix + self.log_edisp_norm_matrix


            
            
            if 'log_irf_norm_matrix_path' in self.config_dict:
                np.save(self.config_dict['log_irf_norm_matrix_path'], self.log_irf_norm_matrix)
            else:
                if self.save_path=="":
                    self.log_irf_norm_matrix_path = 'irf_norm_matrix.npy'
                else:
                    self.log_irf_norm_matrix_path = self.save_path+'/irf_norm_matrix.npy'


                np.save(self.log_irf_norm_matrix_path, self.log_irf_norm_matrix)
                self.log_psf_norm_matrix_path = os.path.abspath(self.log_irf_norm_matrix_path)
    
                print(f"Saving to {self.log_irf_norm_matrix_path}")

            
            if 'log_psf_norm_matrix_path' in self.config_dict:
                self.log_psf_norm_matrix_path = self.config_dict['log_psf_norm_matrix_path']
                np.save(self.config_dict['log_psf_norm_matrix_path'], self.log_psf_norm_matrix)

            else:
                if self.save_path=="":
                    self.log_psf_norm_matrix_path = 'log_psf_norm_matrix.npy'
                else:
                    self.log_psf_norm_matrix_path = self.save_path+'/log_psf_norm_matrix.npy'


                np.save(self.log_psf_norm_matrix_path, self.log_psf_norm_matrix)
                self.log_psf_norm_matrix_path = os.path.abspath(self.log_psf_norm_matrix_path)
                self.config_dict['log_psf_norm_matrix_path'] = self.log_psf_norm_matrix_path
                

            

            if 'log_edisp_norm_matrix_path' in self.config_dict:
                self.log_edisp_norm_matrix_path = self.config_dict['log_edisp_norm_matrix_path']

                np.save(self.config_dict['log_edisp_norm_matrix_path'], self.log_edisp_norm_matrix)
            else:
                if self.save_path=="":
                    self.log_edisp_norm_matrix_path = 'log_edisp_norm_matrix.npy'
                else:
                    self.log_edisp_norm_matrix_path = self.save_path+'/log_edisp_norm_matrix.npy'

                    
                np.save(self.log_edisp_norm_matrix_path, self.log_edisp_norm_matrix)
                self.log_edisp_norm_matrix_path = os.path.abspath(self.log_edisp_norm_matrix_path)
                self.config_dict['log_edisp_norm_matrix_path'] = self.log_edisp_norm_matrix_path


        
        
        self.log_exposure_map = GammaLogExposure(binning_geometry = self.true_binning, 
                                             irfs=self.irf_loglike,)


    def _handle_dm_specifications(self):
        """
        Handles dark matter specifications.
        """

        if 'dark_matter_density_profile' in self.config_dict:
            density_profile_string = self.config_dict['dark_matter_density_profile']
            self.dark_matter_density_profile = dynamic_import(
                'gammabayes.dark_matter.density_profiles',  density_profile_string)
        else:
            density_profile_string = 'Einasto_Profile'
            self.dark_matter_density_profile = Einasto_Profile



        if 'dark_matter_mass' in self.config_dict:
            self.dark_matter_mass = self.config_dict['dark_matter_mass']
        else:
            self.dark_matter_mass = np.nan



        if 'dark_matter_spectral_model' in self.config_dict:
            self.use_dm_model = True
            self.use_dm_channels = False
            dm_spectral_model_string = self.config_dict['dark_matter_spectral_model']

            self.dark_matter_spectral_class = dynamic_import(
                'gammabayes.dark_matter.spectral_models',  dm_spectral_model_string)
            

            self.dark_matter_model = CombineDMComps(
                name=f'DM',

                spectral_class = self.dark_matter_spectral_class,
                spatial_class = self.dark_matter_density_profile,

                axes = self.true_axes,

                irf_loglike = self.irf_loglike, 

                default_spectral_parameters={'mass':self.dark_matter_mass}
                )
            

        elif 'DM_Channels' in self.config_dict:
            self.use_dm_model = False
            self.use_dm_channels = True

            self.DM_Channels = self.config_dict['DM_Channels']

            if 'DM_Annihilation_Ratios' in self.config_dict:
                self.DM_Annihilation_Ratios = self.config_dict['DM_Annihilation_Ratios']
            else:
                self.DM_Annihilation_Ratios = None

            self.custom_ratios_model = CustomDMRatiosModel(
                name='DM',
                channels=self.DM_Channels,
                irf_loglike=self.irf_loglike, 
                spatial_class=self.dark_matter_density_profile,
                axes=self.true_axes, 
                default_spectral_parameters={'mass':self.dark_matter_mass})

        else:
            self.use_dm_model = False
            self.use_dm_channels = False


        if self.use_dm_channels:
            if not('DM' in self.mixture_parameter_specifications):
                warnings.warn(
                    """When using dark matter models by channel, there must be a specifications for the
                    overall signal fraction named 'DM' within the mixture parameter specifications.""", 
                    category=UserWarning)




    def _handle_missing_true_specifications(self):
        """
        Handles missing true parameter specifications.
        """
        if 'true_mixture_fractions' in self.config_dict:
            self.true_mixture_fractions = self.config_dict['prior_parameter_specifications']
        else:
            self.true_mixture_fractions = None


        if 'Nevents_per_job' in self.config_dict:
            self.Nevents_per_job = self.config_dict['Nevents_per_job']
        elif 'Nevents' in self.config_dict:
            self.Nevents_per_job = self.config_dict['Nevents']
        else:
            self.Nevents_per_job = np.nan


        if 'DM_Annihilation_Ratios' in self.config_dict:
            self.DM_Annihilation_Ratios = self.config_dict['DM_Annihilation_Ratios']
        else:
            self.DM_Annihilation_Ratios = None

        if not('marginalisation_bounds' in self.config_dict):
            self.config_dict['marginalisation_bounds'] = None



    def _setup_observational_prior_models(self):
        """
        Sets up observational prior models.
        """


        self.observational_prior_model_names = self.config_dict['observational_prior_models']

        self.observational_prior_models = ['Not Set']*len(self.observational_prior_model_names)
        nested_model_idx_offset = 0

        warnings.filterwarnings("ignore", category=RuntimeWarning)


        for model_idx, model_name in enumerate(self.observational_prior_model_names):

            print(f"Setting up {model_name}")
            
            if not(model_name in ['DM', 'CCR_BKG', 'FG_HS_CCR']):

                prior_model_class = dynamic_import('gammabayes.priors', model_name)
                prior_model = prior_model_class(
                    axes=self.true_axes,
                    irf_loglike=self.irf_loglike)
                
                self.observational_prior_models[model_idx+nested_model_idx_offset] = prior_model

        
            elif model_name=='DM':
                if self.use_dm_model:
                    prior_model = self.dark_matter_model
                    self.observational_prior_models[model_idx+nested_model_idx_offset] = prior_model
                elif self.use_dm_channels:
                    num_channels  = len(self.custom_ratios_model.channels)
                    self.observational_prior_models.extend(['not set channels']*(num_channels-1))
                    self.observational_prior_models[model_idx:model_idx+num_channels] = list(self.custom_ratios_model.values())
                    nested_model_idx_offset+=num_channels-1



            elif model_name=='CCR_BKG':
                ccr_bkg_prior = ObsFluxDiscreteLogPrior(
                    logfunction=self.irf_loglike.log_bkg_CCR, 
                    name='CCR_BKG',
                    axes=self.true_axes, 
                    pointing_dir=self.irf_loglike.pointing_dir)
                
                self.observational_prior_models[model_idx+nested_model_idx_offset] = ccr_bkg_prior


            elif model_name=='FG_HS_CCR':

                true_meshes = np.meshgrid(*self.true_axes, indexing='ij')

                fermi_gaggero_rate_matrix = construct_fermi_gaggero_flux_matrix(binning_geometry = self.true_binning)
                hess_source_rate_matrix = construct_hess_flux_matrix(binning_geometry = self.true_binning,)
                log_CCR_matrix = self.irf_loglike.log_bkg_CCR(energy=true_meshes[0], longitude=true_meshes[1], latitude=true_meshes[2])


                log_bkg_matrix = special.logsumexp([np.log(fermi_gaggero_rate_matrix.value), np.log(hess_source_rate_matrix.value), log_CCR_matrix], axis=0)

                # Pseudo-normalisation to regularise/make calculations more numerically stable
                log_bkg_matrix = log_bkg_matrix - special.logsumexp(log_bkg_matrix)

                fixed_background_interpolator = RegularGridInterpolator(points=self.true_axes, values=np.exp(log_bkg_matrix))


                log_obs_interpolator_bkgs = log_obs_interpolator(fixed_background_interpolator)



                bkg_prior = DiscreteLogPrior(logfunction=log_obs_interpolator_bkgs.log_func, name='FG_HS_CCR',
                            axes=self.true_axes, )
                self.observational_prior_models[model_idx+nested_model_idx_offset] = bkg_prior


            elif model_name=='FG_HS':



                fermi_gaggero_rate_matrix =construct_fermi_gaggero_flux_matrix(energy_axis=self.true_axes[0], longitudeaxis=self.true_axes[1], latitudeaxis=self.true_axes[2],
                                                log_exposure_map=self.log_exposure_map)
                hess_source_rate_matrix = construct_hess_flux_matrix(energy_axis=self.true_axes[0], longitudeaxis=self.true_axes[1], latitudeaxis=self.true_axes[2],
                                                log_exposure_map=self.log_exposure_map)

                fixed_background_interpolator = RegularGridInterpolator(points=self.true_axes, values=fermi_gaggero_rate_matrix+hess_source_rate_matrix)


                log_obs_interpolator_bkgs = log_obs_interpolator(fixed_background_interpolator)



                FG_HS_bkg_prior = DiscreteLogPrior(logfunction=log_obs_interpolator_bkgs.log_func, name='FG_HS',
                            axes=self.true_axes, )
                self.observational_prior_models[model_idx+nested_model_idx_offset] = FG_HS_bkg_prior

        warnings.filterwarnings("default", category=RuntimeWarning)


    def _setup_parameter_specifications(self):
        """
        Sets up parameter specifications for the priors.
        """

        num_obs_priors = len(self.observational_prior_models)

        formatted_prior_parameter_specifications = {}


        for model_idx, obs_prior in enumerate(self.observational_prior_models):

            if obs_prior.name in self.prior_parameter_specifications:
                unique_parameter_spec = ParameterSet(self.prior_parameter_specifications[obs_prior.name])
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







    def _setup_mixture_tree_values(self, values:list=None):
        """
        Sets up the mixture tree values.

        Args:
            values (iterable): Values to set up in the mixture tree. Defaults to None.
        """

        self.mixture_layout = self.config_dict['mixture_layout']

        if values is None:

            unformatted_values = []

            if 'true_mixture_fractions' in self.config_dict:
                self.true_mixture_fractions = self.config_dict['true_mixture_fractions']
            else:
                self.true_mixture_fractions = {}



            if 'DM_Annihilation_Ratios' in self.config_dict:
                self.DM_Annihilation_Ratios = self.config_dict['DM_Annihilation_Ratios']
            else:
                self.DM_Annihilation_Ratios = {}

            values = []
            for source_name, fraction in self.true_mixture_fractions.items():


                values.append(fraction)

                if source_name=='DM':
                    values.extend(list(self.DM_Annihilation_Ratios.values()))

        if len(values) <1:
            values=None


        self.mixture_tree = MTree()
        self.mixture_tree.create_tree(self.mixture_layout, values=values)

        print("__________ MIXTURE TREE STRUCTURE __________")
        print("""note: if 'True' values have been unspecified, 
then the tree node values are the defaults such that if you add them up for their given branch you get 1.\n""")

        print(self.mixture_tree)











        



        



        

        
        


