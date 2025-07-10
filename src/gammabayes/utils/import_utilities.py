import importlib
import requests
import tarfile
import importlib.resources as pkg_resources
from pathlib import Path
from tqdm import tqdm
from io import BytesIO

# Function to dynamically import the desired function
def dynamic_import(module_path: str, object_name: str):
    """
    Dynamically imports a specified object (e.g., function, class) from a given module.

    Args:
        module_path (str): The path to the module from which to import the object.
        object_name (str): The name of the object to import from the module.

    Returns:
        object: The imported object if successful, or None if an error occurred.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the object cannot be found in the module.
    """

    try:
        # Import the module
        module = importlib.import_module(module_path)
        
        # Fetch the function from the module
        func = getattr(module, object_name)
        
        return func
    except (ImportError, AttributeError) as e:
        # Handle the error (module or function not found)
        print(f"Error importing {object_name}: {e}")
        return None


# This assumes that 'package_data' is a subdirectory inside your package
def _get_package_data_directory():
    # Locate the package_data directory within your package
    package_data_dir_path = pkg_resources.files('gammabayes') / 'package_data'
    
    return package_data_dir_path


# Function to download the tar file and unpack it
def download_and_unpack_tar(url, download_dir=None, verify=True):
    # Use package_data directory by default
    if download_dir is None:
        download_dir = _get_package_data_directory()

    download_dir = Path(download_dir)
    
    # Define the tar file path in the download directory
    tar_file_path = download_dir / url.split("/")[-1]
    
    # Check if the file already exists
    if not tar_file_path.exists():
        print(f"Downloading {tar_file_path.name}...")
        
        # Download with progress bar
        response = requests.get(url, stream=True, verify=verify)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB block size for better performance

        with tar_file_path.open('wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as progress_bar:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                progress_bar.update(len(chunk))

        print(f"Download completed: {tar_file_path}")
    else:
        print(f"Tar file already exists at {tar_file_path}")

    # Open the tar file and extract it
    try:
        with tarfile.open(tar_file_path, mode="r:") as tar:
            tar.extractall(path=download_dir)
        print(f"Unpacking completed. Data extracted to {download_dir}")
    except tarfile.ReadError as e:
        try: 
            with tarfile.open(tar_file_path, mode="r:gz") as tar:
                tar.extractall(path=download_dir)
            print(f"Unpacking completed. Data extracted to {download_dir}")
        except tarfile.ReadError as e:
            print(f"Error while extracting the tar file: {e}")
            raise ValueError(f"The file may be corrupted or not a valid .tar file.")
    
    return download_dir