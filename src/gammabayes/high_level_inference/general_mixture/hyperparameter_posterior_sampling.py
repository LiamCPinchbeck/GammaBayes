import numpy as np


from gammabayes.utils import (
    dynamic_import,
    )

from .setup import hl_setup_from_config



class high_level_mixture(hl_setup_from_config):
    """
    High-level mixture model class that extends `hl_setup_from_config` for setting up and simulating mixture models.

    Args:
        hl_setup_from_config (_type_): Base class for setup from configuration.
    """

    def __init__(self, config, **kwargs):
        """
        Initializes the high-level mixture model with the provided configuration.

        Args:
            config (_type_): Configuration settings for the mixture model.
        """

        super().__init__(config=config, **kwargs)


    def simulate(self, Nevents=None, tree_node_values=None):
        """
        Simulates the mixture model to generate true and measured event data.

        Args:
            Nevents (int, optional): Number of events to simulate. Defaults to the configured number of events per job.
            tree_node_values (dict, optional): Values to overwrite in the mixture tree. Defaults to None.

        Returns:
            tuple: True event data and measured event data.
        """

        sample_mixture_tree = self.mixture_tree.copy()
        true_event_data = []

        if not(tree_node_values is None):
            sample_mixture_tree.overwrite(tree_node_values)

        
        if Nevents is None:
            Nevents = self.Nevents_per_job


        leaf_values = list(sample_mixture_tree.leaf_values.values())

        num_samples = np.asarray(leaf_values)*Nevents

        true_event_data = [obs_prior.sample(num_sample) for obs_prior, num_sample in zip(self.observational_prior_models, num_samples)]

        # true_events = true_event_data[0]
        true_events = sum(true_event_data)


        

        measured_event_data = self.irf_loglike.sample(true_events)

        return true_event_data, measured_event_data
    

    def nuisance_marginalisation(self, measured_event_data):
        """
        Performs nuisance parameter marginalisation on the measured event data.

        Args:
            measured_event_data (_type_): The measured event data.

        Returns:
            _type_: Reshaped log marginal results.
        """

        self.discrete_scan_hyperparameter_likelihood = dynamic_import(
            'gammabayes.hyper_inference',  
            self.config_dict['hyper_parameter_scan_class'])



        discrete_scan_hyperparameter_likelihood_instance = self.discrete_scan_hyperparameter_likelihood(
            log_priors=self.observational_prior_models,
            log_likelihood=self.irf_loglike, 
            nuisance_axes = self.true_axes, 
            axes=self.recon_axes,
            prior_parameter_specifications=self.obs_prior_parameter_specifications, 
            log_likelihoodnormalisation=self.log_irf_norm_matrix,
            bounds=self.config_dict['marginalisation_bounds'],
            mixture_fraction_exploration_type='sample')


        reshaped_log_marg_results = discrete_scan_hyperparameter_likelihood_instance.nuisance_log_marginalisation(
            measured_event_data)
        
        return reshaped_log_marg_results

    

