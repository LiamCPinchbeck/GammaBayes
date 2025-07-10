from gammabayes.high_level_inference import High_Level_Analysis
from gammabayes.likelihoods.irfs import IRF_LogLikelihood
from gammabayes import GammaBinning, GammaObs, Parameter, ParameterSet, ParameterSetCollection
from astropy import units as u 
import os, sys, numpy as np, yaml
from multiprocessing.pool import Pool
from tqdm import tqdm
from echidma import EchidmaH5
from echidma.numpy_full_load import EchidmaNumpyLoad


def run_marg():

    config_file_name = sys.argv[1]

    try:
        skip_irf_norm = sys.argv[2] 
    except:
        skip_irf_norm = False
    

    print("\n\n_______ JOB SETUP _______\n")
    analysis_class = High_Level_Analysis.from_config_file(config_file_name, skip_irf_norm=skip_irf_norm)

    marginalisation_method = analysis_class.marginalisation_method
    marginalisation_bounds = analysis_class.marginalisation_bounds


    observation_time = analysis_class.config_dict["observation_times"][0]

    observational_priors = list(analysis_class.observation_cube.observations[0].meta['priors'].values())

    irf_loglike = IRF_LogLikelihood(
        axes                =   analysis_class.recon_binning_geometry.axes,
        dependent_axes      =   analysis_class.true_binning_geometry.axes,
        pointing_dir        =   np.array(analysis_class.config_dict["pointing_directions"][0])*u.deg,
        observation_time    =   u.Quantity(observation_time) if observation_time is not None else None,)


    log_likelihoodnormalisation = analysis_class.irf_log_norm





    marginalisation_class_instance = marginalisation_method(
        log_priors                      =   observational_priors, 
        log_likelihood                  =   irf_loglike, 
        axes                            =   analysis_class.recon_binning_geometry.axes, 
        log_likelihoodnormalisation     =   log_likelihoodnormalisation,
        nuisance_axes                   =   analysis_class.true_binning_geometry.axes, 
        prior_parameter_specifications  =   [ParameterSet(param_set) for param_set in analysis_class.obs_prior_parameter_specifications], 
        bounds                          =   marginalisation_bounds,
    )


    job_id = analysis_class.config_dict['job_id']
    save_path = analysis_class.config_dict['save_path']

    meas_event_data = np.load(analysis_class.config_dict['event_data_path'])

    Nevents_per_job = len(meas_event_data[0])


    num_batches = analysis_class.config_dict['marg_cores']
    batch_size = int(round(Nevents_per_job/num_batches))+1 # +1 is to make sure all events are included


    units = [axis.unit for axis in analysis_class.true_binning_geometry.axes]


    batched_event_data = []
    batch_counter = 0
    for batch_idx, batch in enumerate(range(num_batches)):
        batched_event_data.append([meas_event_data_variable[batch_counter:batch_counter+batch_size]*unit for meas_event_data_variable, unit in zip(meas_event_data, units)])
        batch_counter+=batch_size


    print("\n\n_______ MARGINALISATION _______\n")
    batched_log_marg_results = []

    print("Performing Marginalisation....")
    with Pool(analysis_class.config_dict['marg_cores']) as pool:

        # We specifically use "imap" instead of "imap_unordered" here (for example) due to ease-of-debugging reasons allowing one to relate the given
            # event values to the marginalisation behaviour
        for result in tqdm(pool.imap(marginalisation_class_instance.nuisance_log_marginalisation, batched_event_data), desc="Batches Marginalised", total=num_batches):
            batched_log_marg_results.append(result)


    obs_prior_specs = marginalisation_class_instance.prior_parameter_specifications
    obs_prior_names = [log_prior.name for log_prior in marginalisation_class_instance.log_priors]

    print("\n\nCombining Marginalisation Results....\n")
    num_priors = len(batched_log_marg_results[0])
    log_marg_results = [None]*num_priors
    for batch_idx in range(num_batches):
        for prior_idx in range(num_priors):
            
            batched_results = batched_log_marg_results[batch_idx][prior_idx]



            if log_marg_results[prior_idx] is None:
                log_marg_results[prior_idx] = batched_results
            else:
                log_marg_results[prior_idx] = np.append(log_marg_results[prior_idx], batched_results, axis=0)


    # log_marg_results  = analysis_class.nuisance_marginalisation(measured_event_data = measured_event_data)


    for log_marg_result, prior_name, prior_spec in tqdm(zip(log_marg_results, obs_prior_names, obs_prior_specs), desc="Saving results", total=len(obs_prior_names)):
        print(len(log_marg_result))
        if save_path=="":
            log_marg_filename = f'logmarg_{prior_name}_{job_id}.npy'
        else:
            log_marg_filename = f'{save_path}/logmarg_{prior_name}_{job_id}.npy'


        EchidmaNumpyLoad.save(filename=log_marg_filename, array=log_marg_result, parameter_axes=[prior_spec.dict_of_parameters_by_name])



    print("\n\nJob Finished.")


if __name__=="__main__":

    run_marg()