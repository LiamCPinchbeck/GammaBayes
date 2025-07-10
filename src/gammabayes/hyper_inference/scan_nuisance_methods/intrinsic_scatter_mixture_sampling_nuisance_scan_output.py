import numpy as np, warnings, dynesty, logging, time

from gammabayes.hyper_inference.core.utils import _handle_parameter_specification
from gammabayes.hyper_inference.core.mixture_tree import MTree, MTreeNode
from gammabayes.samplers.sampler_utils import ResultsWrapper
from gammabayes import ParameterSet, ParameterSetCollection
from gammabayes.core.core_utils import update_with_defaults
from itertools import islice

import h5py

from .general_mixture_sampling_nuisance_scan_output import ScanOutput_StochasticTreeMixturePosterior
from scipy.stats import beta
from gammabayes.utils.integration import iterate_logspace_integration, logspace_riemann, logspace_simpson


import numpy as np

def logsubexp(a, b):
    """
    Compute log(e^a - e^b) in a numerically stable way.

    Parameters:
    a : float or np.ndarray
        The logarithm of the first term.
    b : float or np.ndarray
        The logarithm of the second term.

    Returns:
    float or np.ndarray
        The logarithm of (e^a - e^b).
    """
    
    output =  a + np.log1p(-np.exp(b - a))
    output = np.where(np.isnan(output), -np.inf, output)
    output = np.where(np.isinf(output), -np.inf, output)

    return output



class ScanOutput_IntrinsicScatterMixturePosterior(ScanOutput_StochasticTreeMixturePosterior):

    def __init__(self, 
                log_nuisance_marg_results: list[np.ndarray]|np.ndarray,
                mixture_tree: MTree,
                mixture_parameter_specifications:list | dict | ParameterSet,
                prior_parameter_specifications: list | dict | ParameterSet = None,
                scatter_parameter_set:ParameterSet=ParameterSet({}),
                observational_prior_names: list=None,
                shared_parameters: list | dict | ParameterSet = None,
                parameter_meta_data: dict = None,
                event_weights:np.ndarray=None,
                log_nuisance_marg_regularisation: float = 0., # Argument for consistency between classes
                _sampler_results={},
                scatter_likelihood:np.ndarray=0,
                scatter_prior:np.ndarray=0,
                scatter_sigma:float=0.01, # corresponds to 1% multiplicative noise on CCR rate
                CCR_BKG_name:str="CCR_BKG",
                scatter_K_range = np.linspace(0.2, 1.8, 401),
                ):
        self.scatter_sigma = scatter_sigma
        self.scatter_parameter_set = scatter_parameter_set
        self.CCR_BKG_name = CCR_BKG_name
        self.scatter_K_range = scatter_K_range
        
        self.scatter_logprior_values = self.mod_beta_scatter_logprior(self.scatter_K_range)

        self.scatter_logprior_norm = logspace_riemann(self.scatter_logprior_values, self.scatter_K_range)

        self.scatter_logprior_values -=self.scatter_logprior_norm

        super().__init__(log_nuisance_marg_results=log_nuisance_marg_results,
                mixture_tree=mixture_tree,
                mixture_parameter_specifications=mixture_parameter_specifications,
                prior_parameter_specifications=prior_parameter_specifications,
                observational_prior_names=observational_prior_names,
                shared_parameters=shared_parameters,
                parameter_meta_data=parameter_meta_data,
                event_weights=event_weights,
                log_nuisance_marg_regularisation=log_nuisance_marg_regularisation,
                _sampler_results=_sampler_results)
        
        



    

    def mod_beta_scatter_logprior(self, delta, sigma=None, E=None):
        if sigma is None:
            sigma = self.scatter_sigma


        a = 1/2*(1/sigma**2+1)
        b = a
        median = a/(a+b)

        log_prior_values = np.log(median) + beta.logpdf(x=median * delta, a=a, b=b)
        
        return log_prior_values
    

    def calc_scatter_marg_values(self, scatter_parameter_set=None):
        if scatter_parameter_set is None:
            scatter_parameter_set = self.scatter_parameter_set
        pass

    @property
    def scatter_marg_arrays(self):

        if hasattr(self, "_scatter_marg_arrays"):
            return self._scatter_marg_arrays
        
        self._scatter_marg_arrays = self.calc_scatter_marg_values()

        return self._scatter_marg_arrays


    def unsmoothness_penalty(self, arr):
        # Roll the array in different directions to get neighbors
        top_right = np.roll(np.roll(arr, -1, axis=0), 1, axis=1)
        right = np.roll(arr, 1, axis=1)
        bottom_right = np.roll(np.roll(arr, 1, axis=0), 1, axis=1)
        top = np.roll(arr, -1, axis=0)
        bottom = np.roll(arr, 1, axis=0)
        top_left = np.roll(np.roll(arr, -1, axis=0), -1, axis=1)
        left = np.roll(arr, -1, axis=1)
        bottom_left = np.roll(np.roll(arr, 1, axis=0), -1, axis=1)

        # Sum neighbors or apply custom processing
        result = -(top_right-arr)**2-(right-arr)**2-(bottom_right-arr)**2-(top-arr)**2-(bottom-arr)**2-(top_left-arr)**2-(left-arr)**2-(bottom_left-arr)**2 

        return result






    def __scatter_intermediary(self, logmarg_ccr_slice, ccr_mix, base_ln_like):

        # Possible issues with second part not properly going through logspace substraction

        scatter_ln_likes = np.where(self.scatter_K_range-1>=0,
                                    np.logaddexp(
                                        base_ln_like[:, None] -np.log(1+ccr_mix*(self.scatter_K_range-1))[None,:], 
                                        (np.log(ccr_mix*(self.scatter_K_range-1))-np.log(1+ccr_mix*(self.scatter_K_range-1)))[None,:]+logmarg_ccr_slice[:, None]),
                                    logsubexp(
                                        base_ln_like[:, None] -np.log(1+ccr_mix*(self.scatter_K_range-1))[None,:], 
                                        (np.log(ccr_mix*(1-self.scatter_K_range))-np.log(1+ccr_mix*(self.scatter_K_range-1)))[None,:]+logmarg_ccr_slice[:, None])
                                        )

        scatter_ln_likes *= self.event_weights[:, None]

        ln_like = self.scatter_logprior_values[None, :] + scatter_ln_likes

        # ln_like-= np.sqrt(len(self.event_weights))*self.unsmoothness_penalty(ln_like)

        ln_like = logspace_riemann(ln_like, x=self.scatter_K_range, axis=-1)

        return ln_like

    def scatter_ln_likelihood(self, inputs:list|np.ndarray, log_nuisance_marg_results:list[np.ndarray]=None):



        if log_nuisance_marg_results is None:
            log_nuisance_marg_results = self.log_nuisance_marg_results


        base_ln_like = self.single_event_loglike(inputs=inputs, log_nuisance_marg_results=log_nuisance_marg_results)

        # Extract values of parameters
        mixture_fractions = inputs[:self.num_mixes]

        self.mixture_tree.overwrite(mixture_fractions)


        ccr_mix = self.mixture_tree.leaf_values[self.CCR_BKG_name]

        ccr_prior_idx = list(self.mixture_tree.leaf_values).index(self.CCR_BKG_name)

        
        logmarg_ccr_slice = log_nuisance_marg_results[ccr_prior_idx]

        ln_like = self.__scatter_intermediary(logmarg_ccr_slice, ccr_mix, base_ln_like)

        combined_ln_like = np.sum(ln_like)

        return combined_ln_like
