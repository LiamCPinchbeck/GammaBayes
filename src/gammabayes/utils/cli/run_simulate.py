from gammabayes.high_level_inference import High_Level_Analysis
import os, sys, numpy as np, yaml
from multiprocessing.pool import Pool
from tqdm import tqdm
from echidma import EchidmaH5
from echidma.numpy_full_load import EchidmaNumpyLoad
import h5py
from icecream import ic


def run_sim():
    print("\n\n_______ JOB SETUP _______\n")
    analysis_class = High_Level_Analysis.from_config_file(sys.argv[1])

    job_id = analysis_class.config_dict['job_id']
    save_path = analysis_class.config_dict['save_path']

    Nevents_for_job = analysis_class.config_dict['Nevents_per_job']

    print("\n\n_______ SIMULATION _______\n")
    print("\nBatching Event Number...\n")
    batched_event_data = []
    batched_true_event_data = []
    batch_num_events = int(round(Nevents_for_job/analysis_class.config_dict['sim_cores']))

    analysis_class.setup_simulation(Nevents=batch_num_events)
    
    batch_num_events = [batch_num_events]*analysis_class.config_dict['sim_cores']


    print(batch_num_events)

    print("\nSimulating Event Data...\n")
    observation_containers_batches = []
    with Pool(analysis_class.config_dict['sim_cores']) as pool:
        for simulation_output in tqdm(pool.imap_unordered(analysis_class.simulation_observation_set, batch_num_events), total=len(batch_num_events), desc="Batches Simulated"):
            non_zero_bin_data = list(simulation_output.values())[0]['measured_event_data'].nonzero_bin_data[1]
            observation_containers_batches.append(simulation_output)



    total_observation_container      = {}

    for observations in observation_containers_batches:
        for obs_id, observation_container in observations.items():

            if obs_id not in total_observation_container.keys():
                total_observation_container[obs_id] = {}
                total_observation_container[obs_id]["true_event_data"] = {}
                total_observation_container[obs_id]["measured_event_data"] = observation_container["measured_event_data"]
            else:
                total_observation_container[obs_id]["measured_event_data"] = total_observation_container[obs_id]["measured_event_data"] + observation_container["measured_event_data"]
            

            for prior_name, true_event_data in observation_container['true_event_data'].items():
                if prior_name in total_observation_container[obs_id]["true_event_data"].keys():
                    total_observation_container[obs_id]["true_event_data"][prior_name] = total_observation_container[obs_id]["true_event_data"][prior_name] + true_event_data
                else:
                    total_observation_container[obs_id]["true_event_data"][prior_name] = true_event_data




    obs_prior_names = total_observation_container[obs_id]["true_event_data"].keys()
    # Save data into a single HDF5 file

    for obs_id, obs_data in total_observation_container.items():
        h5_file_path = os.path.join(save_path, obs_id+'_event_data.h5')
        print(h5_file_path)
        with h5py.File(h5_file_path, 'w') as h5_file:
            # Create 'true' group
            h5_file.attrs['name'] = obs_id
            combined_true_group = h5_file.create_group('true_event_data')

            for prior_name, prior_event_data in obs_data["true_event_data"].items():

                true_group = combined_true_group.create_group(prior_name)

                true_event_data_set = prior_event_data.to_dict()
                for info_name, info_values in true_event_data_set.items():
                    if info_values is None:
                        info_values = np.nan
                    try:
                        true_group.create_dataset(info_name, data=info_values)
                    except Exception as err:
                        ic(info_name, info_values)
                    
                        raise Exception(err)


            # Save measured data
            measured_group = h5_file.create_group('measured_event_data')

            measured_data_dict = obs_data["measured_event_data"].to_dict()
            for info_name, info_values in measured_data_dict.items():
                if info_values is None:
                    info_values = np.nan
                
                try:
                    measured_group.create_dataset(info_name, data=info_values)
                except Exception as err:
                    ic(info_name, info_values)
                    raise Exception(err)


    print("\n\nJob Finished.")


if __name__=="__main__":
    run_sim()