import os, warnings, shutil, requests, tarfile, subprocess
import importlib.resources as pkg_resources

def check_and_download_temp_hess_data(
    tar_folder_name='hess_dl3_dr1',
    repo_folder_name='hess_ost_paper_material',
    temp_folder_name='temp_HESS_DL3_DR1',
    tar_url='https://www.mpi-hd.mpg.de/hfm/HESS/pages/dl3-dr1/hess_dl3_dr1.tar',
    repo_url='https://github.com/lmohrmann/hess_ost_paper_material.git',
    destination=None,
    subfolder='package_data'
):
    package = 'gammabayes'
    
    # Get the data folder location (use package_data by default)
    if destination is None:
        package_dir = pkg_resources.files(package).joinpath(subfolder)
        data_dir = package_dir
    else:
        data_dir = os.path.abspath(destination)

    # Create a temporary folder where both datasets will be downloaded
    temp_data_dir = os.path.join(data_dir, temp_folder_name)
    
    # Paths where the specific data should be stored within the temp folder
    tar_extracted_dir = os.path.join(temp_data_dir, tar_folder_name)
    repo_extracted_dir = os.path.join(temp_data_dir, repo_folder_name)
    
    # Create the temporary directory if it doesn't exist
    os.makedirs(temp_data_dir, exist_ok=True)

    # Download and extract the tar file
    check_and_download_tar(tar_url, tar_extracted_dir)

    # Clone the repository into a folder
    check_and_download_data_from_github(repo_url, repo_extracted_dir)

def check_and_download_tar(url, extracted_dir):
    tar_path = os.path.join(extracted_dir, 'data.tar')

    # Check if the tar file has already been extracted
    if not os.path.exists(extracted_dir):

        
        # Create the folder if it doesn't exist
        os.makedirs(extracted_dir, exist_ok=True)
        
        # Download the tar file
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

            
            # Extract the tar file
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extracted_dir)


            # Optionally delete the tar file after extraction
            os.remove(tar_path)
        else:
            warnings.warn(f"Failed to download tar file from {url}.")



def check_and_download_data_from_github(repo_url, repo_dir):
    # Check if the folder already exists
    if not os.path.exists(repo_dir):        
        # Clone the repository
        clone_repo(repo_url, repo_dir)
            
    return repo_dir

def clone_repo(repo_url, clone_to):
    # Use git to clone the repository
    result = subprocess.run(['git', 'clone', repo_url, clone_to], 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL,
        check=True)
    if not(result.returncode == 0):
        raise RuntimeError("Git clone failed.")
