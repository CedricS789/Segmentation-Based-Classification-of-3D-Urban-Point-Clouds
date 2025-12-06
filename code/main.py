import os
import sys
from utils import load_point_cloud, RNN_Voxelisation, visualize_point_cloud

# Configuration
VISUALIZE_ANIMATION = True # Set to False to disable voxelization animation

def main():
    # Define the dataset path relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "domfountain_station2_xyz_intensity_rgb.txt")

    print("--- 3D Urban Point Cloud Segmentation Project ---")

    # 1. Load Data
    print("\n[Phase 1] Data Loading")
    # Load only a subset for faster visualization (e.g., 50,000 points)
    df = load_point_cloud(file_path, nrows=50000)
    if df is None:
        return
    
    # Visualize Raw Data
    visualize_point_cloud(df, window_name=f"Raw Data ({len(df):,} points)")

    # 2. Voxelisation
    print("\n[Phase 2] Voxelisation (r-NN)")
    # Using 0.1m radius
    voxelizer = RNN_Voxelisation(df, radius=0.1)

    print(f"Animation enabled: {VISUALIZE_ANIMATION}")
    
    # Run voxelisation
    # NOTE: For checking logic, we limit to first 1000 voxels or process a subset.
    # Processing 41M points iteratively can take hours without C++ optimization.
    # We will test with a sample_size limit first.
    print("Running with sample limit for verification...")
    # Limiting to 5000 voxels to verify it works
    s_voxels = voxelizer.voxelize(sample_size=5000, visualize=VISUALIZE_ANIMATION) 
    
    print("\nVoxelisation Results:")
    print(f"Original Points: {len(df):,}")
    print(f"Super-Voxels:    {len(s_voxels):,}")
    print("\nSample s-voxels:")
    print(s_voxels.head())

    # Visualize Voxelized Data
    visualize_point_cloud(s_voxels, window_name=f"Voxelized Data ({len(s_voxels):,} voxels)")

if __name__ == "__main__":
    main()
