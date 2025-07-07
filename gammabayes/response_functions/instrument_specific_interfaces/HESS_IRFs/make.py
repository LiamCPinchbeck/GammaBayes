import shutil, os
from pathlib import Path
from astropy.io import fits
from astropy.table import Table
import importlib.resources as pkg_resources

from .HESS_files_generation import check_and_download_temp_hess_data

# Function to get observation IDs from the tar dataset
def get_obs_ids(path_hess, debug_run=False):
    obs_id = Table.read(path_hess / "obs-index.fits.gz")["OBS_ID"]
    return obs_id[:3] if debug_run else obs_id

# Function to copy the existing OBS index file to the output directory
def make_obs_index(path_hess, path_out):
    src = path_hess / "obs-index.fits.gz"
    dst = path_out / "obs-index.fits.gz"
    shutil.copyfile(src, dst)

# Function to copy the HDU index file and add background HDU rows
def make_hdu_index(path_hess, path_out, obs_ids):
    path = path_hess / "hdu-index.fits.gz"
    table = Table.read(path)

    for obs_id in obs_ids:
        filename = f"hess_dl3_dr1_obs_id_{obs_id:06d}.fits.gz"
        size = fits.open(path_out / f"data/{filename}")["bkg"].filebytes()
        table.add_row([obs_id, "bkg", "bkg_3d", "data", filename, "bkg", size])

    table.sort(["OBS_ID", "HDU_TYPE"])
    table.write(path_out / "hdu-index.fits.gz", overwrite=True)

# Function to create the background HDU by converting the data to float32
def make_background_hdu(path_bkg_background, obs_id):
    path = path_bkg_background / f"hess_bkg_3d_{obs_id:06d}.fits.gz"
    table = Table.read(path)

    for colname in table.colnames:
        table[colname] = table[colname].astype("float32")

    hdu = fits.BinTableHDU(table)
    hdu.name = "bkg"
    return hdu

# Function to copy existing data file and add background HDU
def make_data_file(path_hess, path_bkg_background, path_out, obs_id):
    path = path_hess / f"data/hess_dl3_dr1_obs_id_{obs_id:06d}.fits.gz"
    hdu_list = fits.open(path)

    hdu_bkg = make_background_hdu(path_bkg_background, obs_id)
    hdu_list.append(hdu_bkg)

    output_path = path_out / f"data/hess_dl3_dr1_obs_id_{obs_id:06d}.fits.gz"
    hdu_list.writeto(output_path, overwrite=True)

# Main function to combine datasets and update index tables
def extract_and_generate_hess_data(temp_dir=None, output_dir=None, debug_run=False):
    """
    Combine hess-dl3-dr1 dataset for Gammapy by merging tar data and repo data.
    
    Parameters:
    - temp_dir: Path to the temporary folder where tar and repo data are stored.
    - output_dir: Path to the folder where the combined dataset will be saved.
    - debug_run: If True, only process a subset of data for testing.
    """

    
    # Default paths for temp and output directories
    package_dir = pkg_resources.files('gammabayes').joinpath('package_data')
    if temp_dir is None:
        temp_dir = package_dir / "temp_HESS_DL3_DR1"
    else:
        temp_dir = Path(temp_dir)
    if output_dir is None:
        output_dir = package_dir / "HESS_DL3_DR1"
    else:
        output_dir = Path(output_dir)




    path_hess = Path(temp_dir) / "hess_dl3_dr1"
    path_bkg_background = Path(temp_dir) / "hess_ost_paper_material" / "background_model" / "data"
    path_out = Path(output_dir)

    # Check if the hdu-index.fits.gz file already exists
    hdu_index_path = path_out / "hdu-index.fits.gz"
    if hdu_index_path.exists():

        return

    check_and_download_temp_hess_data(destination=temp_dir.parent.absolute())

    # Create the output directory if it doesn't exist
    path_out.mkdir(parents=True, exist_ok=True)
    (path_out / "data").mkdir(exist_ok=True)

    obs_ids = get_obs_ids(path_hess, debug_run)

    for obs_id in obs_ids:
        make_data_file(path_hess, path_bkg_background, path_out, obs_id)

    make_obs_index(path_hess, path_out)
    make_hdu_index(path_hess, path_out, obs_ids)


    # Delete the temporary directory after combining files
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


# Script entry point
if __name__ == "__main__":
    # Run the combination script (will use default directories if unspecified)
    combine_datasets()
