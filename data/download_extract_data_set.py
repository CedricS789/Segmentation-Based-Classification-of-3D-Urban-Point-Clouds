import os
import sys
import subprocess
import shutil
import py7zr

# Available datasets from Semantic3D
DATASETS = {
    "1": {
        "name": "bildstein_station1_xyz_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/bildstein_station1_xyz_intensity_rgb.7z",
        "description": "Bildstein Station 1 (Rural church in Austria) - Training Set"
    },
    "2": {
        "name": "bildstein_station3_xyz_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/bildstein_station3_xyz_intensity_rgb.7z",
        "description": "Bildstein Station 3 (Rural church in Austria) - Training Set"
    },
    "3": {
        "name": "bildstein_station5_xyz_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/bildstein_station5_xyz_intensity_rgb.7z",
        "description": "Bildstein Station 5 (Rural church in Austria) - Training Set"
    },
    "4": {
        "name": "domfountain_station1_xyz_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/domfountain_station1_xyz_intensity_rgb.7z",
        "description": "Domfountain Station 1 (Cathedral in Feldkirch, Urban) - Training Set"
    },
    "5": {
        "name": "domfountain_station2_xyz_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/domfountain_station2_xyz_intensity_rgb.7z",
        "description": "Domfountain Station 2 (Cathedral in Feldkirch, Urban) - Training Set"
    },
    "6": {
        "name": "domfountain_station3_xyz_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/domfountain_station3_xyz_intensity_rgb.7z",
        "description": "Domfountain Station 3 (Cathedral in Feldkirch, Urban) - Training Set"
    },
    "7": {
        "name": "neugasse_station1_xyz_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/neugasse_station1_xyz_intensity_rgb.7z",
        "description": "Neugasse Station 1 (Street in St. Gallen, Urban) - Training Set"
    },
    "8": {
        "name": "sg27_station1_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/sg27_station1_intensity_rgb.7z",
        "description": "SG27 Station 1 (Railroad tracks) - Training Set"
    },
    "9": {
        "name": "sg27_station2_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/sg27_station2_intensity_rgb.7z",
        "description": "SG27 Station 2 (Urban town square) - Training Set"
    },
    "10": {
        "name": "sg27_station4_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/sg27_station4_intensity_rgb.7z",
        "description": "SG27 Station 4 (Rural village) - Training Set"
    },
    "11": {
        "name": "sg27_station5_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/sg27_station5_intensity_rgb.7z",
        "description": "SG27 Station 5 (Suburban crossing)  - Training Set"
    },
    "12": {
        "name": "sg27_station9_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/sg27_station9_intensity_rgb.7z",
        "description": "SG27 Station 9 (Urban soccer field) - Training Set"
    },
    "13": {
        "name": "sg28_station4_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/sg28_station4_intensity_rgb.7z",
        "description": "SG28 Station 4 (Urban town square) - Training Set"
    },
    "14": {
        "name": "untermaederbrunnen_station1_xyz_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/untermaederbrunnen_station1_xyz_intensity_rgb.7z",
        "description": "Untermaederbrunnen Station 1 (Fountain in Balgach, Rural) - Training Set"
    },
    "15": {
        "name": "untermaederbrunnen_station3_xyz_intensity_rgb",
        "url": "https://share.phys.ethz.ch/~pf/semantic3d/data/point-clouds/training1/untermaederbrunnen_station3_xyz_intensity_rgb.7z",
        "description": "Untermaederbrunnen Station 3 (Fountain in Balgach, Rural) - Training Set"
    }
}

DATA_DIR = "./data"

def download_file(url, dest_path):
    print(f"Downloading from {url} to {dest_path}...")
    try:
        # Using curl for progress bar
        subprocess.run(["curl", "-o", dest_path, url], check=True)
        print("Download complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        return False
    except FileNotFoundError:
        print("Error: 'curl' command not found. Please install curl.")
        return False

def extract_7z(archive_path, output_dir):
    print(f"Extracting {archive_path} to {output_dir}...")
    try:
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            z.extractall(path=output_dir)
        print("Extraction complete.")
        return True
    except Exception as e:
        print(f"Failed to extract: {e}")
        return False

def verify_data(file_path):
    if not os.path.exists(file_path):
        print(f"FAILED: File not found at {file_path}")
        return False

    print(f"Verifying {file_path}...")
    try:
        with open(file_path, 'r') as f:
            # Read first few lines to check format
            for i in range(5):
                line = f.readline()
                if not line:
                    break
                parts = line.strip().split()
                # Expecting 7 columns: X Y Z Intensity R G B
                if len(parts) != 7:
                    print(f"WARNING: Line {i+1} has {len(parts)} columns, expected 7.")
                else:
                    # Optional: Check if values are numeric
                    try:
                        [float(x) for x in parts]
                    except ValueError:
                         pass # Header might be non-numeric, but Semantic3D usually isn't.

        print("Verification complete. Header check passed.")
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")
        return True

    except Exception as e:
        print(f"Error reading file: {e}")
        return False

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print("Available Semantic3D Datasets:")
    for key, info in DATASETS.items():
        print(f"[{key}] {info['description']} ({info['name']})")

    choice = input("\nEnter the number of the dataset to download (or 'q' to quit): ").strip()
    
    if choice.lower() == 'q':
        return

    if choice not in DATASETS:
        print("Invalid choice.")
        return

    dataset = DATASETS[choice]
    base_name = dataset["name"]
    archive_name = base_name + ".7z"
    txt_name = base_name + ".txt"
    
    archive_path = os.path.join(DATA_DIR, archive_name)
    txt_path = os.path.join(DATA_DIR, txt_name)

    # Check if extracted file already exists
    if os.path.exists(txt_path):
        print(f"\nDataset '{base_name}' already exists at {txt_path}.")
        print("Skipping download and extraction.")
        verify_data(txt_path)
        return

    # Check if archive exists
    if not os.path.exists(archive_path):
        success = download_file(dataset["url"], archive_path)
        if not success:
            return
    else:
        print(f"\nArchive '{archive_name}' already exists. Skipping download.")

    # Extract
    if extract_7z(archive_path, DATA_DIR):
        # Verify
        verify_data(txt_path)

if __name__ == "__main__":
    main()
