import os
import sys
from utils import load_point_cloud, RNN_Voxelisation, visualize_point_cloud
# Configuration
VISUALIZE_ANIMATION = True      # Flag to enable/disable voxelization animation
# Dataset path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(BASE_DIR, "data", "domfountain_station2_xyz_intensity_rgb.txt")
SAMPLE_POINTS = 1000000         # Number of raw points to load
VOXEL_RADIUS = 0.1              # Radius for r-NN voxelization (in meters)
VOXEL_SAMPLE_SIZE = 5000        # Max number of voxels to generate (for testing)

# Visualization Settings
VIZ_ZOOM = 0.3                  # Camera zoom level
VIZ_FRONT = [0.5, -0.86, 0.5]   # Camera front direction
VIZ_LOOKAT = [0, 0, 0]          # Center of the point cloud
VIZ_UP = [0, 0, 1]              # Z-up

def main():
    print("--- 3D Urban Point Cloud Segmentation Project ---")

    # 1. Load Data
    print("\n[Phase 1] Data Loading")
    # 1. Load Data
    print("\n[Phase 1] Data Loading")
    # Load a subset of points for performance optimization during testing
    df = load_point_cloud(FILE_PATH, nrows=SAMPLE_POINTS)
    if df is None:
        return
    
    # Prepare visualization settings
    viz_config = {
        'zoom': VIZ_ZOOM,
        'front': VIZ_FRONT,
        'lookat': VIZ_LOOKAT,
        'up': VIZ_UP
    }

    # Visualize Raw Data
    visualize_point_cloud(df, window_name=f"Raw Data ({len(df):,} points)", viz_settings=viz_config)

    # 2. Voxelisation
    print(f"\n[Phase 2] Voxelisation (r-NN) r={VOXEL_RADIUS}m")
    # Using 0.1m radius
    voxelizer = RNN_Voxelisation(df, radius=VOXEL_RADIUS)

    print(f"Animation enabled: {VISUALIZE_ANIMATION}")
    
    # Run voxelisation
    # Limit voxel generation to a sample size for verification purposes.
    # Full processing of 41M points requires C++ optimization or extended execution time.
    print("Running with sample limit for verification...")
    # Generate a limited number of voxels to verify functionality
    s_voxels = voxelizer.voxelize(sample_size=VOXEL_SAMPLE_SIZE, visualize=VISUALIZE_ANIMATION) 
    
    print("\nVoxelisation Results:")
    print(f"Original Points: {len(df):,}")
    print(f"Super-Voxels:    {len(s_voxels):,}")
    print("\nSample s-voxels:")
    print(s_voxels.head())

    # Visualize Voxelized Data
    visualize_point_cloud(s_voxels, window_name=f"Super-Voxels (Averaged Data) ({len(s_voxels):,} voxels)", viz_settings=viz_config)

if __name__ == "__main__":
    main()
