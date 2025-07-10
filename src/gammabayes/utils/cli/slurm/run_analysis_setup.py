import os, sys, yaml, time, numpy as np
from tqdm import tqdm
from importlib.metadata import version
from gammabayes.high_level_inference import High_Level_Analysis
import h5py
from gammabayes import GammaObs

from matplotlib import pyplot as plt
from astropy import units as u






def analysis_setup():

    with open(sys.argv[1], 'r') as file:
        stem_config_file = yaml.safe_load(file)

    print("\n\n")
    analysis_class = High_Level_Analysis.from_config_dict(stem_config_file)


    pointing_dirs = [np.array(pointing)*u.deg for pointing in analysis_class.config_dict["pointing_directions"]]
    observation_times = [u.Quantity(obs_time) for obs_time in analysis_class.config_dict["observation_times"]]
    print(pointing_dirs)

    stem_path = f"data/{stem_config_file['stem_identifier']}"
    job_outputs_path = f"{stem_path}/job_outputs"


    simulation_job_data_stem_path = f"data/{stem_config_file['stem_identifier']}/job_data/simulation_data"

    marginalisation_job_data_stem_path = f"data/{stem_config_file['stem_identifier']}/job_data/marginalisation_data"
    os.makedirs(marginalisation_job_data_stem_path, exist_ok=True)



    measured_event_data_by_observation = {}

    for root, dirs, files in os.walk(simulation_job_data_stem_path):
        for file in files:


            if file.endswith('_event_data.h5'):
                # Load the data from the HDF5 file
                relative_filepath = os.path.relpath(os.path.join(root, file), os.getcwd())

                with h5py.File(relative_filepath, 'r') as h5_file:
                    # Load the measured data
                    observation_name = h5_file.attrs['name']

                    print(observation_name)
                    binned_data = h5_file['measured_event_data']['binned_data'][:]
                    print("Num events: ", np.sum(binned_data))
                    print(h5_file['measured_event_data'].get('observation_time_unit', [None])[()])
                    print(h5_file['measured_event_data'].get('observation_time', [None])[()])

                    
                    file_observation_data = GammaObs.load_from_dict(
                        info_dict={"name" :observation_name,
                        "pointing_dir": h5_file['measured_event_data']['pointing_dir'][:],
                        "pointing_dir_unit": h5_file['measured_event_data'].get('pointing_dir_unit')[()],
                        "observation_time": h5_file['measured_event_data'].get('observation_time', None)[()],
                        "observation_time_unit": h5_file['measured_event_data'].get('observation_time_unit', None)[()],
                        "binned_data": binned_data,
                        "binning_geometry":analysis_class.recon_binning_geometry}
                        )


                    if observation_name not in measured_event_data_by_observation:
                        measured_event_data_by_observation[observation_name] = file_observation_data
                    else:
                        measured_event_data_by_observation[observation_name] = measured_event_data_by_observation[observation_name] + file_observation_data


    unique_events_by_observation = {}
    for observation_id, observation_data in measured_event_data_by_observation.items():

        unique_events_by_observation[observation_id] = len(observation_data.nonzero_coordinate_data[0])

    total_unique_events = sum([num_events for num_events in unique_events_by_observation.values()])

    print("total unique events: ", total_unique_events)
    print("unique_events_by_observation: ", list(unique_events_by_observation.items()))

    job_counter = 0
    for (observation_id, observation_data), pointing_dir, observation_time in zip(measured_event_data_by_observation.items(), pointing_dirs, observation_times):

        num_unique_measured_events = unique_events_by_observation[observation_id]
        num_jobs_for_obs = int(round(stem_config_file['num_marg_jobs']*(unique_events_by_observation[observation_id]/total_unique_events)))



        batch_size = int(round(num_unique_measured_events/num_jobs_for_obs))
        print(observation_id, batch_size)

        # echidma_configs = [{}*config_file['num_priors']]

        batch_counter = 0
        nonzero_event_coord_tuple, nonzero_event_counts = observation_data.nonzero_bin_data

        for marginalisation_job_id in tqdm(range(1, num_jobs_for_obs+1)):
            marginalisation_job_file_stem = f"{marginalisation_job_data_stem_path}/{job_counter+marginalisation_job_id}"


            os.makedirs(marginalisation_job_file_stem, exist_ok=True)

            marginalisation_job_config_file = stem_config_file.copy()


            batched_event_weights = nonzero_event_counts[batch_counter:batch_counter+batch_size]

            batched_unique_events = [nonzero_coords[batch_counter:batch_counter+batch_size] for nonzero_coords in nonzero_event_coord_tuple]
            batch_counter+=batch_size


            # gamma_obs_for_batch = GammaObs(
            #     binning_geometry=observation_data.binning_geometry,
            #     name = observation_data.name+f"_{marginalisation_job_id}",
            #     observation_time=observation_time,
            #     irf_loglike=observation_data.irf_loglike,
            #     pointing_dir=pointing_dir,
            #     energy = batched_unique_events[0],
            #     lon = batched_unique_events[1],
            #     lat = batched_unique_events[2],
            #     event_weights = batched_event_weights,
            #     )


            np.save(f"{marginalisation_job_file_stem}/meas_event_weights.npy", batched_event_weights )
            event_data_path = f"{marginalisation_job_file_stem}/meas_event_data.npy"

            np.save(event_data_path, batched_unique_events)

            # Dealing with annoying yaml
            pointing_dir_value = list(np.round(pointing_dir.value, decimals = 4))
            for pointing_idx, pointing in enumerate(pointing_dir_value):
                if pointing==-0.0:
                    pointing_dir_value[pointing_idx] = 0.0

                pointing_dir_value[pointing_idx] = float(pointing_dir_value[pointing_idx])
            
            marginalisation_job_config_file['event_data_path'] = event_data_path

            marginalisation_job_config_file['save_path'] = marginalisation_job_file_stem
            marginalisation_job_config_file['seed'] = marginalisation_job_id
            marginalisation_job_config_file['job_id'] = marginalisation_job_id
            marginalisation_job_config_file['Nevents_per_job'] = batch_size
            marginalisation_job_config_file['pointing_directions'] = [pointing_dir_value]

            if hasattr(observation_time, "unit"):
                obs_time = observation_time.to_string()
            else:
                obs_time = observation_time

            marginalisation_job_config_file['observation_times'] = [obs_time]

            
            marginalisation_job_config_file['time'] = time.strftime("%H:%M:%S")
            marginalisation_job_config_file['date'] = time.strftime("%Y/%m/%d")
            marginalisation_job_config_file['gammabayes_vers'] = version('gammabayes')
            marginalisation_job_config_file['echidma_vers'] = version('echidma')
            


            with open(f'{marginalisation_job_file_stem}/job_config.cfg', 'w') as outfile:
                yaml.safe_dump(marginalisation_job_config_file, 
                        outfile, 
                        default_flow_style=False,
                        sort_keys=False)
                

        job_counter+=marginalisation_job_id
                
                


    #### Create marginalisation job file
    marg_job_str = f"""#!/bin/bash
    #SBATCH --job-name={stem_config_file['stem_identifier']}_marginalisation                         # Job name
    #SBATCH --output=data/{stem_config_file['stem_identifier']}/job_outputs/output_{stem_config_file['stem_identifier']}_marg_%A_%a.out    # Output and error log
    #SBATCH --error=data/{stem_config_file['stem_identifier']}/job_outputs/error_{stem_config_file['stem_identifier']}_marg_%A_%a.err      # Error log
    #SBATCH --ntasks=1                                                                          # Run a single task
    #SBATCH --mem={stem_config_file['marg_mem']}
    #SBATCH --cpus-per-task={stem_config_file['marg_cores']}                                          # Number of CPU cores per task
    #SBATCH --array=1-{stem_config_file['num_marg_jobs']}                                                 # Array of jobs
    #SBATCH --time={stem_config_file['marg_time']}                   
    #SBATCH --mail-type=ALL
    #SBATCH --mail-user=progressemail1999@gmail.com



    # Set the path to your configuration files
    CONFIG_DIR={marginalisation_job_data_stem_path}


    # Run your Python script with the configuration file
    srun python run_marg.py ${{CONFIG_DIR}}/${{SLURM_ARRAY_TASK_ID}}/job_config.cfg"""


    with open(stem_path+"/marg_job.sh", "w") as myFile:
        for line in marg_job_str:
            myFile.write(line)


    #### Create combination job file

    comb_job_str = f"""#!/bin/bash
    #SBATCH --job-name={stem_config_file['stem_identifier']}_combine                                 # Job name
    #SBATCH --output={job_outputs_path}/output_{stem_config_file['stem_identifier']}_comb_%A_%a.out    # Output and error log
    #SBATCH --error={job_outputs_path}/error_{stem_config_file['stem_identifier']}_comb_%A_%a.err      # Error log
    #SBATCH --ntasks=1                                                                          # Run a single task
    #SBATCH --mem={stem_config_file['comb_mem']}
    #SBATCH --cpus-per-task={stem_config_file['comb_cores']}                                          # Number of CPU cores per task
    #SBATCH --time={stem_config_file['comb_time']}                   
    #SBATCH --mail-type=ALL
    #SBATCH --mail-user=progressemail1999@gmail.com



    # Set the path to your configuration files
    CONFIG_DIR={stem_path}


    # Run your Python script with the configuration file
    srun python run_combine.py {stem_path}/stem_config.cfg"""


    with open(stem_path+"/combine_job.sh", "w") as myFile:
        for line in comb_job_str:
            myFile.write(line)


    print("\nAnalysis Setup Complete.\n\n")



if __name__=="__main__":
    analysis_setup()