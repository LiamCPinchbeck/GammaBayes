from .scan_reweighting_class import ScanReweighting
import dynesty, time, numpy as np
from gammabayes.utils import iterate_logspace_integration
from scipy import special
from gammabayes.samplers.sampler_utils import ResultsWrapper
from gammabayes import (
    Parameter, ParameterSet, 
    apply_dirichlet_stick_breaking_direct, update_with_defaults
)
from gammabayes.priors import DiscreteLogPrior
import h5py


class StochasticReweighting(ScanReweighting):
    """
    Implements stochastic reweighting to adjust the weights of samples 
    obtained from Bayesian inference processes, such as nested sampling, 
    in response to new priors or likelihoods. This class is designed with 
    flexibility in mind, aiming for compatibility with various sampling 
    libraries, though it is initially tailored for use with the `dynesty` 
    nested sampling library.

    Note:
        This class is in an experimental phase and currently operates with 
        limited efficiency. Optimizations and enhancements are planned for 
        future updates.

    Methods marked with `NotImplementedError` are inherited from a parent class 
    and are not applicable in this specific context. Their implementation is 
    intentionally omitted to avoid misuse within the framework of stochastic 
    reweighting.

    """

    
    def __init__(self, _sampler_results, *args, **kwargs):
        """
        Initializes the StochasticReweighting instance, setting up the necessary parameters
        and configurations for reweighting.

        Inherits initialization from `ScanReweighting`, allowing for custom setup through
        passed arguments and keyword arguments.

        Parameters:
            measured_events (EventData): The event data to be analyzed.
            
            log_likelihood (DiscreteLogLikelihood): An instance representing the log-likelihood function.
            
            log_proposal_prior (DiscreteLogPrior): An instance representing the log of the proposal prior.
            
            log_target_priors (list[DiscreteLogPrior]): A list of instances representing the log of the target priors.
            
            nuisance_axes (list[np.ndarray], optional): Nuisance parameter axes (true energy, longitude and latitude).
            
            mixture_parameter_specifications (dict | ParameterSet, optional): Specifications for mixture model parameters.
            
            marginalisation_bounds (list[(str, float)], optional): Bounds for nuisance parameter marginalization.
            
            bounding_percentiles (list[float], optional): Percentiles used to calculate bounding for the nuisance 
            parameter marginalisation if 'marginalisation_bounds' is not given.
            
            bounding_sigmas (list[float], optional): Sigma levels used to calculate bounding for the nuisance 
            parameter marginalisation if 'marginalisation_bounds' is not given.
            
            logspace_integrator (callable, optional): A callable for log-space integration (multi dimensions).
            
            prior_parameter_specifications (dict | list[ParameterSet], optional): Parameter sets for the priors.
            
            reweight_batch_size (int, optional): Batch size for reweighting operation.
            
            no_priors_on_init (bool, optional): If True, no errors are raised if priors are not supplied on 
            initialization. 
            
            log_prior_normalisations (np.ndarray, optional): Normalisation constants for the log priors.
        """

        super().__init__(*args, **kwargs)

    # To ensure a mix of methods aren't used unless I create equivalent methods
    def reweight_single_param_combination(self, *args, **kwargs):
        raise NotImplementedError
    
    def reweight_single_prior(self, *args, **kwargs):
        raise NotImplementedError
    
    def scan_reweight(self, *args, **kwargs):
        raise NotImplementedError
    
    def select_scan_output_posterior_exploration_class(self, *args, **kwargs):
        raise NotImplementedError
    

    def init_posterior_exploration(self, *args, **kwargs):
        raise NotImplementedError
    
    def run_posterior_exploration(self, *args, **kwargs):
        raise NotImplementedError
    


    def prior_transform(self, u, mixture_parameters, num_hyper_axes, num_mixes):
        """
        Transforms uniform variables `u` into the parameter space defined by the prior. 
        This method adjusts the provided uniform variables for both the mixture components 
        and the hyperparameters of the model.

        Args:
            u (np.ndarray): Array of uniform variables to be transformed.
            
            mixture_parameters (list): Parameters defining the mixture model components.
            
            num_hyper_axes (int): Number of hyperparameters in the model.
            
            num_mixes (int): Number of mixture components.

        Returns:
            np.ndarray: Transformed variables aligned with the model's parameter space.
        """


        # We can just use the individual parameter transforms to pump out the relevant values
        for _mix_idx in range(num_mixes):
            u[_mix_idx] = mixture_parameters[_mix_idx].transform(u[_mix_idx])
        for _hyper_idx in range(num_hyper_axes):
            u[num_mixes + _hyper_idx] = self.parameters[_hyper_idx].transform(u[num_mixes + _hyper_idx])

        return u
    

    def evaluate_prior_on_batch(self, batch_proposal_posterior_samples, 
                                       prior, 
                                       prior_spectral_params_vals,
                                       prior_spatial_params_vals):
        """
        Evaluates the prior probability for a batch of samples. This method facilitates 
        the reweighting process by allowing batch processing of sample evaluations under 
        new prior assumptions.

        Args:
            batch_proposal_posterior_samples (np.ndarray): Samples to evaluate.
            
            prior (DiscreteLogPrior): Prior distribution to use for evaluation.
            
            prior_spectral_params_vals (dict): Spectral parameters for the prior.
            
            prior_spatial_params_vals (dict): Spatial parameters for the prior.

        Returns:
            np.ndarray: Evaluated prior probabilities for the batch of samples.
        """

        prior_values = prior(
                        batch_proposal_posterior_samples[:,0],
                        batch_proposal_posterior_samples[:,1],
                        batch_proposal_posterior_samples[:,2],
                        spectral_parameters  = {spec_key:batch_proposal_posterior_samples[:,0]*0.+spec_val for spec_key, spec_val in prior_spectral_params_vals.items()}, 
                        spatial_parameters   = {spat_key:batch_proposal_posterior_samples[:,0]*0.+spat_val for spat_key, spat_val in prior_spatial_params_vals.items()})

        return prior_values
    


    def ln_likelihood(self, inputs, log_proposal_evidence_values, proposal_posterior_samples, num_mixes):

        """
        Computes the log-likelihood for a given set of inputs against the proposal 
        evidence values and posterior samples. This function is critical for the 
        reweighting process, as it determines the likelihood of the data under 
        the new model parameters.

        Args:
            inputs (np.ndarray): Inputs including mixture weights and hyperparameter values.
            
            log_proposal_evidence_values (list): Log evidence values from the original sampling.
            
            proposal_posterior_samples (list): Posterior samples from the original sampling.
            
            num_mixes (int): Number of mixture components in the reweighting model.

        Returns:
            float: The computed log-likelihood value.
        """

        
        mixture_weights = inputs[:num_mixes]
        hyper_values    = inputs[num_mixes:]

        # Calculating the log of the mixture fractions based on a Dirichlet stick breaking process
        log_mixture_values_array = np.log([apply_dirichlet_stick_breaking_direct(mixture_weights, depth=prior_id) for prior_id in range(self._num_target_priors)])



        # Mapping the output parameters back to their respective priors
        prior_params_dicts    = [{'spectral_parameters':{}, 'spatial_parameters':{}} for target_prior_idx in range(self._num_target_priors)]
        for hyper_parameter, hyper_parameter_value in zip(self.parameters, hyper_values):
            prior_params_dicts[hyper_parameter.prior_id][hyper_parameter.parameter_type][hyper_parameter.name] = hyper_parameter_value


        # Calculating the normalisations for the priors for the given prior parameters
        target_prior_norms  = np.asarray([target_prior.normalisation(**prior_params_dict) for target_prior, prior_params_dict in zip(self.log_target_priors, prior_params_dicts)])

        # Below line is done for in the case that the prior has a zero probability for the entire parameter space
            # e.g. dark matter continuous spectrum with a mass of the lowest observable energy value
        target_prior_norms[np.where(np.isinf(target_prior_norms))] = 0


        # We finally calculate the hyperparameter likelihood values for each event
        ln_likelihood_values = []
        for log_evidence, samples in zip(log_proposal_evidence_values, proposal_posterior_samples):

            num_samples = len(samples[:,0])
            target_prior_values     = np.empty(shape=(num_samples,self.num_priors,))

            proposal_prior_values = \
                self.evaluate_prior_on_batch(batch_proposal_posterior_samples=samples, 
                                    prior=self.log_proposal_prior, 
                                    prior_spectral_params_vals={},
                                    prior_spatial_params_vals={})
            

            #********************************************************
            #********************************************************
            # Note that this is the part we need to optimise
                # Takes more than 0.1s to evaluate
            for prior_id, target_prior in enumerate(self.log_target_priors):
                
                
                target_prior_values[:, prior_id] = \
                    self.evaluate_prior_on_batch(batch_proposal_posterior_samples=samples, 
                                    prior=target_prior, 
                                    prior_spectral_params_vals=prior_params_dicts[prior_id]['spectral_parameters'],
                                    prior_spatial_params_vals=prior_params_dicts[prior_id]['spatial_parameters'],)

            #********************************************************
            #********************************************************
            
            # The above calculates all the above values so that we can do this reweighting sum
            added_target_prior_values = special.logsumexp(target_prior_values+log_mixture_values_array[None, :]-target_prior_norms, axis=1)

            ln_likelihood_value = log_evidence - np.log(num_samples) + special.logsumexp(added_target_prior_values-proposal_prior_values+self.proposal_prior_ln_norm)

            ln_likelihood_values.append(ln_likelihood_value)


        # Then we multiply the likelihoods for each event together
        ln_like = np.sum(ln_likelihood_values)

            
        return ln_like
    

    def init_reweighting_sampler(self, 
                                 log_target_priors: list[DiscreteLogPrior] = None,
                                 reweight_batch_size: int = None, 
                                 mixture_parameter_specifications: list | dict | ParameterSet = None, 
                                 prior_parameter_specifications: list[ParameterSet] = None, 
                                 sampling_class=dynesty.NestedSampler, 
                                 log_proposal_evidence_values: list = None, 
                                 proposal_posterior_samples: list = None,
                                 **kwargs):
        """
        Initializes the reweighting sampler with the specified configuration. This method 
        prepares the sampler for execution, setting up target priors, parameter specifications, 
        and other necessary components based on the inputs.

        Args:
            log_target_priors (list[DiscreteLogPrior], optional): Target priors for reweighting.
            
            reweight_batch_size (int, optional): Batch size for the reweighting process.
            
            mixture_parameter_specifications (list | dict | ParameterSet, optional): Specifications for mixture parameters.
            
            prior_parameter_specifications (list[ParameterSet], optional): Specifications for prior parameters.
            
            sampling_class (class, optional): The sampling class to be used, with `dynesty.NestedSampler` as the default.
            
            log_proposal_evidence_values (list, optional): Log evidence values from the proposal distribution.
            
            proposal_posterior_samples (list, optional): Posterior samples from the proposal distribution.
            
            **kwargs: Additional keyword arguments for the sampler.

        Returns:
            An initialized sampler instance ready for reweighting operations.
        """



        #**************************************************************************************
        #**************************************************************************************
        # The following is essentially to allow the user to specify parameters,
            # but if not given the defaults or previously given values are used
            # and if the ones stored within the class have not been initialised then
            # they are assign the ones given
        if log_target_priors is None:
             log_target_priors = self.log_target_priors
            
        if self.log_target_priors is None:
            self.log_target_priors = log_target_priors

        if reweight_batch_size is None:
            reweight_batch_size = self.reweight_batch_size
        self.reweight_batch_size = reweight_batch_size
        
        if mixture_parameter_specifications is None:
            mixture_parameter_specifications = self.mixture_parameter_specifications

        if self.mixture_parameter_specifications is None:
            self.mixture_parameter_specifications = mixture_parameter_specifications

        if log_proposal_evidence_values is None:
            log_proposal_evidence_values = self.log_proposal_evidence_values

        if self.log_proposal_evidence_values is None:
            self.log_proposal_evidence_values = log_proposal_evidence_values


        if proposal_posterior_samples is None:
            proposal_posterior_samples = self.proposal_posterior_samples
        
        if self.proposal_posterior_samples is None:
            self.proposal_posterior_samples = proposal_posterior_samples

        if not(prior_parameter_specifications is None):
            self.prior_parameter_specifications = prior_parameter_specifications

        #**************************************************************************************
        #**************************************************************************************
        # Now the actual bit of the code



        # The following is done both to clean the inputs but also make it so that the mixture 
            # parameters are stored in an int indexable object so that they can be easily
            # acessed within the likelihood and prior transform
        mixture_parameters = []

        for mix_name, mixture_parameter_specifications in mixture_parameter_specifications.items():
            mixture_parameter = Parameter(mixture_parameter_specifications)
            if not('name' in mixture_parameter):
                mixture_parameter['name'] = mix_name
                
            mixture_parameters.append(mixture_parameter)


        # We do something similar for the mixture parameters, but combining all the given parameters
            # from all the priors into a single list. 

        self.parameters = []
        for prior_id, single_prior__parameter_specifications in enumerate(prior_parameter_specifications):
            for parameter_name, single_parameter_specification in single_prior__parameter_specifications.items():
                parameter               = Parameter(single_parameter_specification)
                parameter['name']       = parameter_name
                parameter['prior_id']   = prior_id
                self.parameters.append(parameter)


        # So we know how many indices are mixture parameters vs prior parameters
        num_hyper_axes = sum([len(parameterset) for parameterset in prior_parameter_specifications])
        num_mixes = len(mixture_parameters)

        # For the sampler to know to store however many values in u
        ndim = num_hyper_axes + num_mixes

        self.num_priors = len(self.log_target_priors)


        # We only need to calculate this once, so we do it here then use it in the above
        self.proposal_prior_ln_norm = self.log_proposal_prior.normalisation()

    
        # Theoretically as long as you specify a sampler with the same names for it's kwargs
            # you should be able to use whatever you want. But at the moment it's specifically 
            # tailored for dynesty
        self.sampler = sampling_class(loglikelihood=self.ln_likelihood, 
                                               prior_transform=self.prior_transform, 
                                               ndim=ndim, 
                                               logl_kwargs = {
                                                   'log_proposal_evidence_values':log_proposal_evidence_values, 
                                                   'proposal_posterior_samples':proposal_posterior_samples,
                                                   'num_mixes':num_mixes
                                               },
                                               ptform_kwargs={
                                                   'mixture_parameters':mixture_parameters,
                                                   'num_hyper_axes':num_hyper_axes,
                                                   'num_mixes':num_mixes,
                                                   },
                                               **kwargs)
        
        return self.sampler
    

    def run_nested(self,*args, **kwargs):
        """
        Executes the nested sampling process with the initialized reweighting sampler. This method 
        is a wrapper around the sampler's `run_nested` method, facilitating the actual execution 
        of the reweighting process.

        Args:
            *args: Positional arguments for the nested sampling execution.
            **kwargs: Keyword arguments for the nested sampling execution.

        Returns:
            The output from the nested sampling run, typically including updated weights and samples.
        """
        

        self.run_nested_output = self.sampler.run_nested(*args, **kwargs)

        return self.run_nested_output
    
    @property
    def posterior_exploration_results(self):
        """
        Provides access to the results of the posterior exploration process, typically 
        including the reweighted samples and associated statistical information.
        Allows common code API with scanning classes

        Returns:
            The results object from the nested sampling library used, containing the 
            reweighted posterior distribution and other relevant metrics.
        """

        return self.results
    
    @property
    def results(self):
        """
        Provides access to the results of the posterior exploration process, typically 
        including the reweighted samples and associated statistical information. Allows
        common API with the nested sampler.

        Returns:
            The results object from the nested sampling library used, containing the 
            reweighted posterior distribution and other relevant metrics.
        """

        return self.sampler.results
    

    def _pack_data(self, h5f=None, file_name=None, reduce_mem_consumption: bool = True):

        if h5f is None:
            h5f = h5py.File(file_name, 'w-')



        if hasattr(self, 'log_proposal_evidence_values'):
            h5f.create_dataset('log_proposal_evidence_values', data=self.log_proposal_evidence_values)


        if self.proposal_posterior_samples is not None:
            # Save log_nuisance_marg_results
            proposal_posterior_samples_group = h5f.create_group("proposal_posterior_samples")
            for result_idx, result in enumerate(self.proposal_posterior_samples):
                proposal_posterior_samples_group.create_dataset(str(result_idx), data=result)
        
        h5f.attrs['log_marginalisation_regularisation'] = self.log_marginalisation_regularisation
        
        if self.nuisance_axes is not None:
            nuisance_axes_group = h5f.create_group('nuisance_axes') 
            for nuisance_axis_idx, nuisance_axis in enumerate(self.nuisance_axes):
                nuisance_axis_dataset = nuisance_axes_group.create_dataset(f"{nuisance_axis_idx}", data=nuisance_axis)

        if self.prior_parameter_specifications is not None:
            prior_param_set_group = h5f.create_group('prior_param_set')

            for prior_idx, (prior, single_prior_param_set) in enumerate(zip(self.log_target_priors, self.prior_parameter_specifications)):
                if type(single_prior_param_set) == ParameterSet:
                    single_prior_param_group = prior_param_set_group.create_group(prior.name)
                    single_prior_param_group = single_prior_param_set.pack(h5f=single_prior_param_group)
        
        
        
        if self.mixture_parameter_specifications is not None:

            mixture_parameter_specifications_group = h5f.create_group('mixture_parameter_specifications')

            mixture_parameter_specifications_group = self.mixture_parameter_specifications.pack(h5f=mixture_parameter_specifications_group)

        bound_types = [bound[0] for bound in self.bounds]
        bound_values = [bound[1] for bound in self.bounds]

        dt = h5py.string_dtype(encoding='utf-8', length=max(len(s) for s in bound_types))

        string_ds = h5f.create_dataset('bound_types', (len(bound_types),), dtype=dt)
        string_ds[:] = bound_types

        h5f.create_dataset('bound_values', data=bound_values)



        if hasattr(self.sampler, 'results'):
            sampler_group=  h5f.create_group('sampler_results')
            results = self.sampler.results
            
            # Save samples
            if hasattr(results, 'samples'):
                sampler_group.create_dataset('samples', data=np.array(results.samples))
            
            # Save log weights
            if hasattr(results, 'logwt'):
                sampler_group.create_dataset('logwt', data=np.array(results.logwt))
            
            # Save log likelihoods
            if hasattr(results, 'logl'):
                sampler_group.create_dataset('logl', data=np.array(results.logl))
            
            # Save evidence information, if available
            if hasattr(results, 'logz'):
                sampler_group.create_dataset('logz', data=np.array(results.logz))
            
            if hasattr(results, 'logzerr'):
                sampler_group.create_dataset('logzerr', data=np.array(results.logzerr))

            if hasattr(results, 'information'):
                sampler_group.create_dataset('information', data=np.array(results.information))

            if hasattr(results, 'nlive'):
                sampler_group.attrs['nlive'] = int(results.nlive)

            if hasattr(results, 'niter'):
                sampler_group.attrs['niter'] = int(results.niter)

            if hasattr(results, 'eff'):
                sampler_group.attrs['eff'] = float(results.niter)

    def pack_data(self, h5f=None, file_name=None, save_log_marg_results=False):
        return self._pack_data(h5f=h5f, file_name=file_name, save_log_marg_results=save_log_marg_results)
    

    def save(self, file_name):
        """
        Saves the class data to an HDF5 file.

        Args:
            file_name (str): The name of the file to save the data to.
        """
        h5f = self.pack_data(file_name=file_name)
        h5f.close()



    @classmethod
    def load(cls, file_name=None):
        """
        Loads the class data from an HDF5 file.

        Args:
            file_name (str): The path to the HDF5 file to load.

        Returns:
            An instance of the class reconstructed from the file.
        """
        init_params = {}
        with h5py.File(file_name, 'r') as h5f:
            # Load datasets if they exist

            init_params = {
                '_log_marginalisation_regularisation': h5f.attrs['log_marginalisation_regularisation'],
            }

            if 'log_proposal_evidence_values' in h5f:
                init_params['log_proposal_evidence_values'] = np.asarray(h5f['log_proposal_evidence_values'])

            if 'proposal_posterior_samples' in h5f:
                init_params['proposal_posterior_samples'] = np.asarray(h5f['proposal_posterior_samples'])

            if 'nuisance_axes' in h5f:
                init_params['nuisance_axes'] = [np.asarray(h5f['nuisance_axes'][str(idx)]) for idx in range(len(h5f['nuisance_axes']))]



            # Load mixture_parameter_specifications
            if "mixture_parameter_specifications" in h5f:
                mixture_param_group = h5f["mixture_parameter_specifications"]
                # Assuming ParameterSet class has a corresponding load or similar method
                init_params["mixture_parameter_specifications"] = ParameterSet.load(mixture_param_group)
            else:
                init_params["mixture_parameter_specifications"] = None
            
            # Load prior_parameter_specifications
            prior_parameter_specifications = []
            if "prior_parameter_specifications" in h5f:
                prior_parameters_group = h5f["prior_parameter_specifications"]
                for prior_idx in sorted(prior_parameters_group, key=int):
                    prior_group = prior_parameters_group[str(prior_idx)]
                    # Assuming ParameterSet.load is capable of handling the loading process
                    prior_parameters = ParameterSet.load(prior_group)
                    prior_parameter_specifications.append(prior_parameters)

            init_params["prior_parameter_specifications"] = prior_parameter_specifications

            _sampler_results = {}
            if 'sampler_results' in h5f:
                sampler_group = h5f['sampler_results']  # Access the group where the sampler data is stored
                
                # Load samples
                if 'samples' in sampler_group:
                    _sampler_results['samples'] = np.array(sampler_group['samples'])
                
                # Load log weights
                if 'logwt' in sampler_group:
                    _sampler_results['logwt'] = np.array(sampler_group['logwt'])
                
                # Load log likelihoods
                if 'logl' in sampler_group:
                    _sampler_results['logl'] = np.array(sampler_group['logl'])
                
                # Load evidence information, if available
                if 'logz' in sampler_group:
                    _sampler_results['logz'] = np.array(sampler_group['logz'])
                if 'logzerr' in sampler_group:
                    _sampler_results['logzerr'] = np.array(sampler_group['logzerr'])

                if 'information' in sampler_group:
                    _sampler_results['information'] = np.array(sampler_group['information'])

                if 'nlive' in sampler_group:
                    _sampler_results['nlive'] = int(sampler_group['nlive'])

                if 'niter' in sampler_group:
                    _sampler_results['niter'] = int(sampler_group['niter'])

                if 'eff' in sampler_group:
                    _sampler_results['eff'] = float(sampler_group['eff'])

            _sampler_results = ResultsWrapper(_sampler_results)

            init_params['_sampler_results'] = _sampler_results


            # Load bound types and values
            if 'bound_types' in h5f and 'bound_values' in h5f:
                bound_types = list(h5f['bound_types'])
                bound_values = h5f['bound_values'][()]
                init_params['marginalisation_bounds'] = list(zip(bound_types, bound_values))

            init_params['no_priors_on_init'] = True


            # Reconstruct the class instance
            instance = cls(
                **init_params
            )
            
        return instance



