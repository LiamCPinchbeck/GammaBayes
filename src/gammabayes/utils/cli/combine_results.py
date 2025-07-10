import sys, time, os
from gammabayes.high_level_inference import High_Level_Analysis
from gammabayes.hyper_inference import ScanOutput_StochasticTreeMixturePosterior
from dynesty.pool import Pool as DyPool
from dynesty import NestedSampler


from echidma import EchidmaMemMap, EchidmaH5, ConfigManager
from echidma.numpy_full_load import EchidmaNumpyLoad

import numpy as np


def run_combine():
    config_file = sys.argv[1]
    analysis_class = High_Level_Analysis.from_config_file(config_file)
    print(f"Number of cores allocated: {analysis_class.config_dict['comb_cores']}")


    log_marg_results = []
    meas_event_weights = []


    obs_prior_names = [prior.name for prior in analysis_class.observation_cube.observations[0].meta["priors"].values()]
    stem_marg_data_path = f"data/{analysis_class.config_dict['stem_identifier']}/job_data/marginalisation_data"

    print("\n\nAccessing data....")
    file_stems = []
    for prior_name in obs_prior_names:
        file_stem = f"logmarg_{prior_name}"
        file_stems.append(file_stem)
        temp_echidma_memmap, event_weight_for_results = EchidmaNumpyLoad.find_and_create_from_config_file(
                    file_stem=file_stem,
                    directory=stem_marg_data_path,
                    cache_size=121,
                    find_secondary = {'stem':'meas_event_weights', 'ext':'.npy', 'loader':np.load}
                    )
                
        log_marg_results.append(temp_echidma_memmap)


        meas_event_weights = [item for sublist in event_weight_for_results.values() for item in sublist]



    meas_event_weights = np.array(meas_event_weights)

    print(f"Number of event weights: {len(meas_event_weights)}")
    print(f"Number of unique events: {len(log_marg_results[0])}")
        

    print("\n\nInstantiating sampling class....")

    hyper_mix_instance = ScanOutput_StochasticTreeMixturePosterior(
        mixture_tree = analysis_class.mixture_tree,
        log_nuisance_marg_results=log_marg_results,
        mixture_parameter_specifications= analysis_class.mixture_parameter_specifications,
        prior_parameter_specifications = analysis_class.obs_prior_parameter_specifications,
        shared_parameters = analysis_class.shared_parameter_specifications,
        event_weights = meas_event_weights
    )


    hyper_loglike = hyper_mix_instance.ln_likelihood
    hyper_prior_transform = hyper_mix_instance.prior_transform
    ndim = hyper_mix_instance.ndim



    print("\n\nInitiating sampler....")

    with DyPool(analysis_class.config_dict['comb_cores'], 
                loglike=hyper_loglike, 
                prior_transform=hyper_prior_transform) as pool:
            
        hyper_mix_instance_sampler = NestedSampler(
            pool.loglike, pool.prior_transform, 
            pool=pool, queue_size=analysis_class.config_dict['comb_cores'],
            ndim=ndim, **analysis_class.config_dict['dynesty_kwargs']['initialisation_kwargs'])
        print("\nSampling...")
        
        hyper_mix_instance_sampler.run_nested(**analysis_class.config_dict['dynesty_kwargs']['run_kwargs'])


    print("\nSampling Completed.")


    sampling_results = hyper_mix_instance_sampler.results


    sampling_results_samples= sampling_results.samples
    sampling_results_logwts= sampling_results.logwt


    print("\nSaving samples and sample weights...")
    np.save(time.strftime(f'data/{analysis_class.config_dict['stem_identifier']}/{analysis_class.config_dict['stem_identifier']}_posterior_samples_%m|%d_%H:%M'), sampling_results_samples)
    np.save(time.strftime(f'data/{analysis_class.config_dict['stem_identifier']}/{analysis_class.config_dict['stem_identifier']}_posterior_logweights_%m|%d_%H:%M'), sampling_results_logwts)

    print("\nScript Completed.\n\n")


if __name__=="__main__":
    run_combine()