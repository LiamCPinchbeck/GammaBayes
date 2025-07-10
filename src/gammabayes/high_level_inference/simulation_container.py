from gammabayes import GammaObs, GammaObsCube, GammaLogExposure, GammaBinning
from gammabayes.priors import DiscreteLogPrior, SourceFluxDiscreteLogPrior, ObsFluxDiscreteLogPrior
from gammabayes.likelihoods.irfs import IRF_LogLikelihood
from gammabayes.hyper_inference import MTree
from gammabayes.utils import iterate_logspace_integration
from tqdm import tqdm
import numpy as np, copy, os, sys
from matplotlib import pyplot as plt
from astropy import units as u
from icecream import ic
from scipy import special

class SimulationContainer:

    def __init__(self, priors:list[DiscreteLogPrior] | list[SourceFluxDiscreteLogPrior], 
                 irf_loglikes, 
                 mixture_tree,
                 true_binning_geometry, 
                 recon_binning_geometry,
                 log_exposures: GammaLogExposure | list[GammaLogExposure], 
                 Nevents: int=1, 
                 observation_times= 0.5*u.hr, 
                 observation_meta=None,
                 pointing_dirs=None,
                 rescale_priors=True):
        self.true_binning_geometry = true_binning_geometry
        self.recon_binning_geometry = recon_binning_geometry
        self.mixture_tree = mixture_tree
        self.priors = priors
        self.log_exposures = log_exposures
        self.irf_loglikes = irf_loglikes
        self.pointing_dirs = pointing_dirs
        self.observation_times = observation_times
        self.Nevents = Nevents

        combined_log_exposure = self.setup_log_exposures()


        if rescale_priors:

            self.rescale_source_flux_priors(mixture_tree=self.mixture_tree, 
                                            Nevents=self.Nevents, 
                                            true_binning_geometry=true_binning_geometry,
                                            combined_log_exposure=combined_log_exposure)
            


        self._setup_observation_containers()
        self._setup_observation_priors()


    def setup_log_exposures(self):

        if isinstance(self.pointing_dirs, np.ndarray):
            if self.pointing_dirs.ndim==1:
                self.pointing_dirs = [self.pointing_dirs]
        
        if isinstance(self.observation_times, (u.Quantity, float)):
            self.observation_times = [self.observation_times]*len(self.log_exposures)
        elif isinstance(self.observation_times, (np.ndarray, list)):
            if len(self.observation_times)!=len(self.log_exposures):
                raise ValueError("Length of observation times is not single value or same length as log_exposures.")


        if isinstance(self.log_exposures, GammaLogExposure):
            init_log_exposure = self.log_exposures
            self.log_exposures = []
            for pointing_dir, obs_time in self.pointing_dirs, self.observation_times:
                pt_exposure = init_log_exposure
                pt_exposure.pointing_dir = pointing_dir
                pt_exposure.observation_time = obs_time
                self.log_exposures.append(pt_exposure)




        self.combined_log_exposure = GammaLogExposure(
            binning_geometry=self.true_binning_geometry, 
            log_exposure_map=special.logsumexp([
                log_exposure.log_exposure_map for log_exposure in self.log_exposures
                ], axis=0))
        

        return self.combined_log_exposure





    def calc_Nevents(self, 
                     true_binning_geometry: GammaBinning = None, 
                     combined_log_exposure:GammaLogExposure=None,
                     observation_times:np.ndarray[u.Quantity]=None,
                     pointing_dirs:np.ndarray[u.Quantity]=None):

        if pointing_dirs is None:
            pointing_dirs = self.pointing_dirs
        if observation_times is None:
            observation_times = self.observation_times


        if true_binning_geometry is None:
            true_binning_geometry = self.true_binning_geometry

        if combined_log_exposure is None:
            combined_log_exposure = self.combined_log_exposure

        event_counts = {}

        for prior in self.priors:
            if isinstance(prior, SourceFluxDiscreteLogPrior):
                prior.log_exposure_map = combined_log_exposure

                log_event_count = prior.log_normalisation()
            

            elif isinstance(prior, ObsFluxDiscreteLogPrior):

                log_event_count = -np.inf
                for pointing_dir, observation_time in zip(pointing_dirs, observation_times):
                    log_event_count_for_pt = prior.log_normalisation(
                                            pointing_dir=pointing_dir,
                                            observation_time=observation_time
                                            )
                    log_event_count = np.logaddexp(log_event_count, log_event_count_for_pt)


            elif isinstance(prior, DiscreteLogPrior):
                
                log_event_count = -np.inf

                for pointing_dir, observation_time in zip(pointing_dirs, observation_times):
                    log_event_count_for_pt = prior.log_normalisation()
                    log_event_count = np.logaddexp(log_event_count, log_event_count_for_pt)


            event_counts[prior.name] = int(round(np.exp(log_event_count)))

        return event_counts


    def rescale_source_flux_priors(self, mixture_tree:MTree, Nevents:dict[str:int], 
                                   true_binning_geometry: GammaBinning = None, 
                                   combined_log_exposure:GammaLogExposure=None):
        
        # TODO: #TechDebt
        unadjusted_event_counts = self.calc_Nevents(true_binning_geometry=true_binning_geometry, 
                                                    combined_log_exposure=combined_log_exposure, 
                                                    observation_times=self.observation_times)
        

        leaf_values = mixture_tree.leaf_values


        events_by_prior = {prior_name:self.Nevents*leaf_value for prior_name, leaf_value in leaf_values.items()}


        event_scaling_factors = {}


        # TODO: #TechDebt
        for prior_idx, (prior_name, num_events) in enumerate(events_by_prior.items()):

            if unadjusted_event_counts[prior_name]==0.:
                scaling_factor = 1
            elif not(np.isinf(np.log(num_events))): #Preventing underflow/neginf issues
                scaling_factor = num_events/unadjusted_event_counts[prior_name]
            else:
                scaling_factor = 1/(unadjusted_event_counts[prior_name]*self.Nevents)

            event_scaling_factors[prior_name] = scaling_factor


            self.priors[prior_idx].rescale(np.log(scaling_factor))



        self.rescaled = True



    
        
    def _setup_observation_containers(self):

                    
        
        if isinstance(self.irf_loglikes, IRF_LogLikelihood) or isinstance(self.irf_loglikes, float):
            initial_loglike = self.irf_loglikes

            self.irf_loglikes = []
            for pointing_dir in self.pointing_dirs:
                pt_irf_loglike = initial_loglike
                pt_irf_loglike.pointing_dir = pointing_dir
                self.irf_loglikes.append(pt_irf_loglike)

        elif isinstance(self.irf_loglikes, (np.ndarray, list)):
            if len(self.irf_loglikes)!=len(self.log_exposures):
                raise ValueError("Length of irf loglikelihoods is not single value or same length as log_exposures.")



        # TODO: #TechDebt
        self.observation_containers = {f"Observation_{obs_idx}":{
            "true_binning_geometry":self.true_binning_geometry, 
            "recon_binning_geometry":self.recon_binning_geometry, 
            "pointing_dir": self.pointing_dirs[obs_idx],
            "observation_time": self.observation_times[obs_idx],
            "irf_loglike": self.irf_loglikes[obs_idx],
            "log_exposure_map": self.log_exposures[obs_idx],
            } for obs_idx in range(len(self.log_exposures))}





    def _setup_observation_priors(self):
        for observation_name in self.observation_containers.keys():
            priors_for_obs = []
            for prior in self.priors:
                if isinstance(prior, SourceFluxDiscreteLogPrior):
                    prior.log_exposure_map = self.observation_containers[observation_name]["log_exposure_map"]
                    priors_for_obs.append(prior)
                elif isinstance(prior, ObsFluxDiscreteLogPrior):
                    prior.pointing_dir = self.observation_containers[observation_name]["pointing_dir"]
                    prior.observation_time = self.observation_containers[observation_name]["observation_time"]
                    priors_for_obs.append(prior)
                else:
                    priors_for_obs.append(prior)


            self.observation_containers[observation_name]['priors'] = priors_for_obs



    def simulate_true_observation(self, 
                                  log_exposure_map: GammaLogExposure, 
                                  priors, 
                                  pointing_dir,
                                  observation_time,
                                  mixture_tree:MTree, 
                                  tree_node_values=None, 
                                  **kwargs) -> dict[GammaObs]:



        sample_mixture_tree = mixture_tree.copy()

        if not(tree_node_values is None):
            sample_mixture_tree.overwrite(tree_node_values)


        true_event_data = {}

        for prior in priors:
            if hasattr(prior, "log_exposure_map"):
                if isinstance(prior, SourceFluxDiscreteLogPrior):
                    log_exposure_map.pointing_dir = pointing_dir
                    log_exposure_map.observation_time = observation_time
                    prior.log_exposure_map = log_exposure_map

                elif isinstance(prior, ObsFluxDiscreteLogPrior):
                    prior.log_exposure_map.pointing_dir = pointing_dir
                    prior.log_exposure_map.observation_time = observation_time

            if hasattr(prior, "pointing_dir"):
                prior.pointing_dir = pointing_dir

            if hasattr(prior, "observation_time"):
                prior.observation_time = observation_time
            prior_samples = prior.sample()

            true_event_data[prior.name] = prior_samples
            


        return true_event_data
    


    def simulation_observation_set(self, print_progress=False):

        simulation_obs_containers  = copy.deepcopy(self.observation_containers)
        data_to_iterate_over = simulation_obs_containers.items()
        if print_progress:
            data_to_iterate_over = tqdm(data_to_iterate_over)
        
        for observation_name, observation_info in data_to_iterate_over:
            observation_info['irf_loglike'].pointing_dir = observation_info['pointing_dir']

            true_event_data = self.simulate_true_observation(mixture_tree=self.mixture_tree, **observation_info)
            simulation_obs_containers[observation_name]["true_event_data"] = true_event_data

            combined_true_event_data = sum(list(true_event_data.values()))



            measured_event_data = observation_info['irf_loglike'].sample(combined_true_event_data)
            
            simulation_obs_containers[observation_name]["measured_event_data"] = measured_event_data

        return simulation_obs_containers



    def peek(self, count_scaling='linear', *args, **kwargs):


        true_event_data = []
        measured_event_data = []

        if "figsize" not in kwargs:
            kwargs["figsize"] = (12, 12)




        for observation_name, observation_info in self.observation_containers.items():

            if "true_event_data" in observation_info:
                true_event_data.append(sum(list(observation_info["true_event_data"].values())))

            if "measured_event_data" in observation_info:
                measured_event_data.append(observation_info["measured_event_data"])

        combined_true_event_data = sum(true_event_data)
        combined_measured_event_data = sum(measured_event_data)


        fig, ax = plt.subplots(nrows=2, ncols=2, *args, **kwargs)

        subplot_kwargs = kwargs

        subplot_kwargs["figsize"] = (subplot_kwargs["figsize"][0], subplot_kwargs["figsize"][1]/2)

        combined_true_event_data.peek(axs=ax[0], count_scaling=count_scaling, **kwargs)
        combined_measured_event_data.peek(axs=ax[1], count_scaling=count_scaling, **kwargs)








