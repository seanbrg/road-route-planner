import kagglehub
import os
import shutil
from pathlib import Path


def setup_massachusetts_data():
    # Define local data directory
    local_data_dir = Path("./data/massachusetts_roads")

    # Check if data already exists locally to avoid redundant moves
    if local_data_dir.exists() and any(local_data_dir.iterdir()):
        print(f"Data already exists at: {local_data_dir.absolute()}")
        return local_data_dir

    print("Downloading dataset from Kaggle...")
    # This downloads to the default kagglehub cache
    cache_path = kagglehub.dataset_download("balraj98/massachusetts-roads-dataset")

    # Create the local directory
    local_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Moving files to {local_data_dir}...")
    # Move files from cache to local /data folder
    for item in os.listdir(cache_path):
        s = os.path.join(cache_path, item)
        d = os.path.join(local_data_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    print("Download and local setup complete.")
    return local_data_dir


if __name__ == "__main__":
    path = setup_massachusetts_data()
    print("Final path to dataset files:", path.absolute())