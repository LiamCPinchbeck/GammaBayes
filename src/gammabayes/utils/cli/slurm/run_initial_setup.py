import os, sys, yaml, time
from tqdm import tqdm
from importlib.metadata import version
from gammabayes.high_level_inference import High_Level_Analysis



def initial_setup():
    with open(sys.argv[1], 'r') as file:
        config_file = yaml.safe_load(file)

    # Make job data folder

    os.makedirs("data", exist_ok=True)

    config_file['stem_identifier'] = config_file['stem_identifier']+time.strftime("_%H%M%S")

    stem_path = os.getcwd()+f"/data/{config_file['stem_identifier']}"
    job_outputs_path = f"{stem_path}/job_outputs"
    os.makedirs(stem_path, exist_ok=True)
    os.makedirs(job_outputs_path, exist_ok=True)


    job_data_stem_path = f"data/{config_file['stem_identifier']}/job_data"
    os.makedirs(job_data_stem_path, exist_ok=True)

    simulation_job_data_stem_path = f"data/{config_file['stem_identifier']}/job_data/simulation_data"
    os.makedirs(simulation_job_data_stem_path, exist_ok=True)



    config_file['save_path'] = stem_path

    print("\n\n")
    analysis_class = High_Level_Analysis.from_config_dict(config_file)


    config_file['log_irf_norm_matrix_path'] = analysis_class.config_dict.get("log_irf_norm_matrix_path")
    config_file['log_psf_norm_matrix_path'] = analysis_class.config_dict.get("log_psf_norm_matrix_path")
    config_file['log_edisp_norm_matrix_path'] = analysis_class.config_dict.get("log_edisp_norm_matrix_path")

    config_file['dynesty_kwargs']['run_kwargs']['checkpoint_file'] = f"{stem_path}/dynesty.save"

    if 'setup_time' not in config_file:
        config_file['setup_time'] = '01:00:00'

    if 'setup_mem' not in config_file:
        config_file['setup_mem'] = config_file['comb_mem']



    with open(f'{stem_path}/stem_config.cfg', 'w') as outfile:
        yaml.safe_dump(config_file, 
                    outfile, 
                    default_flow_style=False,
                    sort_keys=False)

    # echidma_configs = [{}*config_file['num_priors']]

    for simulation_job_id in tqdm(range(1, config_file['num_sim_jobs']+1)):
        simulation_job_file_stem = f"{simulation_job_data_stem_path}/{simulation_job_id}"
        os.makedirs(simulation_job_file_stem, exist_ok=False)

        simulation_job_config_file = config_file.copy()

        simulation_job_config_file['save_path'] = simulation_job_file_stem
        simulation_job_config_file['seed'] = simulation_job_id
        simulation_job_config_file['job_id'] = simulation_job_id
        simulation_job_config_file['Nevents_per_job'] = int(round(simulation_job_config_file['Nevents']/config_file['num_sim_jobs']))

        
        simulation_job_config_file['time'] = time.strftime("%H:%M:%S")
        simulation_job_config_file['date'] = time.strftime("%Y/%m/%d")
        simulation_job_config_file['gammabayes_vers'] = version('gammabayes')
        

        with open(f'{simulation_job_file_stem}/job_config.cfg', 'w') as outfile:
            yaml.safe_dump(simulation_job_config_file, 
                    outfile, 
                    default_flow_style=False,
                    sort_keys=False)
            


    #### Create simulation job file
    sim_job_str = f"""#!/bin/bash
    #SBATCH --job-name={config_file['stem_identifier']}_simulation                         # Job name
    #SBATCH --output=data/{config_file['stem_identifier']}/job_outputs/output_{config_file['stem_identifier']}_sim_%A_%a.out    # Output and error log
    #SBATCH --error=data/{config_file['stem_identifier']}/job_outputs/error_{config_file['stem_identifier']}_sim_%A_%a.err      # Error log
    #SBATCH --ntasks=1                                                                          # Run a single task
    #SBATCH --mem={config_file['sim_mem']}
    #SBATCH --cpus-per-task={config_file['sim_cores']}                                          # Number of CPU cores per task
    #SBATCH --array=1-{config_file['num_sim_jobs']}                                                 # Array of jobs
    #SBATCH --time={config_file['sim_time']}                   
    #SBATCH --mail-type=ALL
    #SBATCH --mail-user=progressemail1999@gmail.com



    # Set the path to your configuration files
    CONFIG_DIR={simulation_job_data_stem_path}


    # Run your Python script with the configuration file
    srun python run_simulate.py ${{CONFIG_DIR}}/${{SLURM_ARRAY_TASK_ID}}/job_config.cfg"""


    with open(stem_path+"/sim_job.sh", "w") as myFile:
        for line in sim_job_str:
            myFile.write(line)




    #### Create analysis setup job file
    sim_job_str = f"""#!/bin/bash
    #SBATCH --job-name={config_file['stem_identifier']}_analysis_setup                         # Job name
    #SBATCH --output=data/{config_file['stem_identifier']}/job_outputs/output_{config_file['stem_identifier']}_anal_setup_%A_%a.out    # Output and error log
    #SBATCH --error=data/{config_file['stem_identifier']}/job_outputs/error_{config_file['stem_identifier']}_anal_setup_%A_%a.err      # Error log
    #SBATCH --ntasks=1                                                                          # Run a single task
    #SBATCH --mem={config_file['setup_mem']}
    #SBATCH --cpus-per-task=1                                          # Number of CPU cores per task
    #SBATCH --time={config_file['setup_time']}                   
    #SBATCH --mail-type=ALL
    #SBATCH --mail-user=progressemail1999@gmail.com



    # Run your Python script with the configuration file
    srun python run_analysis_setup.py {stem_path}/stem_config.cfg"""


    with open(stem_path+"/analysis_setup_job.sh", "w") as myFile:
        for line in sim_job_str:
            myFile.write(line)




    print("\nSimulation Setup Complete.\n\n")

if __name__=="__main__":
    initial_setup()