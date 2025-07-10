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
from icecream import ic



from .setup import High_Level_Setup
from gammabayes.high_level_inference.simulation_container import SimulationContainer
from gammabayes.hyper_inference import DiscreteAdaptiveScan
from gammabayes.hyper_inference import ScanOutput_StochasticTreeMixturePosterior


class High_Level_Analysis(High_Level_Setup):


    # Inherits __init__ from setup class

    def setup_simulation(self, Nevents, *args, **kwargs):
        self.sim_container = SimulationContainer(
            # Presuming that the same sources are considered during all the observations
            priors=list(self.observation_cube.observations[0].meta['priors'].values()), 
            irf_loglikes=self.observation_cube.irf_loglikes, 
            mixture_tree=self.mixture_tree, true_binning_geometry=self.true_binning_geometry,
            recon_binning_geometry = self.recon_binning_geometry,
            log_exposures=self.observation_cube.log_exposures,
            observation_times=self.observation_cube.observation_times,
            pointing_dirs=self.observation_cube.pointing_dirs,
            Nevents=Nevents
            )


    def simulate_true_observation(self, 
                                  log_exposure_map: GammaLogExposure, 
                                  priors, 
                                  pointing_dir,
                                  observation_time,
                                  Nevents=None,
                                  mixture_tree:MTree=None, 
                                  tree_node_values=None,
                                  **kwargs):
        
        if mixture_tree is None:
            mixture_tree = self.mixture_tree

        if not(hasattr(self, 'sim_container')):
            self.setup_simulation(Nevents=Nevents)


        true_obs_data = self.sim_container.simulate_true_observation(
            log_exposure_map = log_exposure_map, 
            priors=priors,
            pointing_dir=pointing_dir,
            observation_time=observation_time,
            mixture_tree=mixture_tree,
            tree_node_values=tree_node_values,
            **kwargs)

        return true_obs_data



    def simulation_observation_set(self, Nevents=None):

        if not(hasattr(self, 'sim_container')):
            self.setup_simulation(Nevents=Nevents)
        
        obs_containers = self.sim_container.simulation_observation_set()

        return obs_containers
    

    @property
    def simulate(self):
        return self.simulation_observation_set()


    def _setup_nuisance_marg(self, fov_irf_norm=None):

        if fov_irf_norm is None:
            fov_irf_norm = self.irf_log_norm



    def run_nuisance_marg(self, observation_containers, fov_irf_norm=None, 
                          marginalisation_method=None, 
                          marginalisation_bounds=None, *args, **kwargs):
        if marginalisation_method is None:
            marginalisation_method = self.marginalisation_method

        if marginalisation_bounds is None:
            marginalisation_bounds = self.marginalisation_bounds

        if fov_irf_norm is None:
            fov_irf_norm = self.irf_log_norm


        all_log_marg_results = []

        for observation_data in observation_containers.values():


            measured_event_data = observation_data["measured_event_data"]

            fov_irf_norm.pointing_dir = observation_data['pointing_dir']

            fov_irf_norm._refresh_buffer_window()

            marginalisation_class = marginalisation_method(
                log_priors=observation_data['priors'], 
                log_likelihood=observation_data['irf_loglike'], 
                axes=observation_data['recon_binning_geometry'].axes, 
                log_likelihoodnormalisation=fov_irf_norm,
                nuisance_axes=observation_data['true_binning_geometry'].axes, 
                prior_parameter_specifications=[ParameterSet(param_set) for param_set in self.obs_prior_parameter_specifications], 
                bounds = marginalisation_bounds
                )


            log_marg_results_for_obs = marginalisation_class.nuisance_log_marginalisation(measured_event_data)


            all_log_marg_results.append(log_marg_results_for_obs)

        return all_log_marg_results
    

    def combine_log_marg_results(self, disconnected_log_marg_results):
        combined_log_marg_results = disconnected_log_marg_results[0]

        for obs_prior_log_marg_results in disconnected_log_marg_results[1:]:
            for prior_idx, log_marg_result in enumerate(obs_prior_log_marg_results):
                combined_log_marg_results[prior_idx] = np.append(combined_log_marg_results[prior_idx], log_marg_result, axis=0)

        self.combined_log_marg_results = combined_log_marg_results

        return combined_log_marg_results


    def setup_hyperparam_analysis(self, 
                                  combined_log_marg_results:list,
                                  observation_containers:dict, 
                                  hyperspace_exploration_method=ScanOutput_StochasticTreeMixturePosterior, 
                                  *args, **kwargs):


        obs_info_list = list(observation_containers.values())

        meas_event_weights = np.concatenate([obs_container['measured_event_data'].nonzero_bin_data[1] for obs_container in obs_info_list])

        # obs_prior_names = [prior.name for prior in obs_info_list[0]['priors']]

        meas_event_weights = np.array(meas_event_weights)

        print(f"Number of event weights: {len(meas_event_weights)}")
            

        print("\n\nInstantiating sampling class....")

        self.hyperspace_sampler = hyperspace_exploration_method(
            mixture_tree = self.mixture_tree,
            log_nuisance_marg_results=combined_log_marg_results,
            mixture_parameter_specifications= self.mixture_parameter_specifications,
            prior_parameter_specifications = self.obs_prior_parameter_specifications,
            shared_parameters = self.shared_parameter_specifications,
            event_weights = meas_event_weights,
        )

        self.hyperspace_sampler.initiate_exploration(**kwargs)


    def run_hyper(self, *args, **kwargs):
        self.hyperspace_sampler.run_exploration(*args, **kwargs)

        return self.hyperspace_sampler.sampler



    @property
    def hyper_ln_like(self):
        return self.hyperspace_sampler.ln_likelihood

    @property
    def hyper_prior_transform(self):
        return self.hyperspace_sampler.prior_transform

    @property
    def hyper_prior(self):
        raise NotImplemented()




    def peek_event_data(self, *args, **kwargs):
        pass

    def print_marg_results_summary(self, *args, **kwargs):
        pass

    def peek_hyperparameter_results(self, *args, **kwargs):
        pass