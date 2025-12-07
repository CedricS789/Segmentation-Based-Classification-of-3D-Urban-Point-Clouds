import os
import sys
from utils import load_point_cloud, RNN_Voxelisation, visualize_point_cloud, save_dataframe, save_point_cloud_ply
# Configuration
VISUALIZE_ANIMATION = True      # Flag to enable/disable voxelization animation
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(BASE_DIR, "data", "domfountain_station2_xyz_intensity_rgb.txt")
MAX_SAMPLE_POINTS = 5000000        # Number of raw points to load. Set to None to load all points.
VOXEL_RADIUS = 1               # Radius for r-NN voxelization (in meters)


# Visualization Settings
VIZ_ZOOM = 0.3                  # Camera zoom level
VIZ_FRONT = [0.5, -0.86, 0.5]   # Camera front direction
VIZ_LOOKAT = [0, 0, 0]          # Center of the point cloud
VIZ_UP = [0, 0, 1]              # Z-up

# Output Files Settings (To save a small amount of data in a CSV)
SAVE_INTERMEDIATES = True       # Flag to save intermediate dataframes to CSV
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

def main():
    print("--- 3D Urban Point Cloud Segmentation Project ---")

    # Create output directory if needed
    if SAVE_INTERMEDIATES and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load Data
    print("\n[Phase 1] Data Loading")
    # Load a subset of points for performance optimization during testing
    df = load_point_cloud(FILE_PATH, nrows=MAX_SAMPLE_POINTS)
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
    print("Displaying Raw Data...")
    visualize_point_cloud(df, window_name=f"Raw Data ({len(df):,} points)", viz_settings=viz_config)

    # Save Intermediate Raw Data
    if SAVE_INTERMEDIATES:
        save_dataframe(df, f"raw_data_{len(df)}.csv", OUTPUT_DIR, None)
        save_point_cloud_ply(df, f"raw_data_{len(df)}.ply", OUTPUT_DIR, None)

    # Voxelisation
    print(f"\n[Phase 2] Voxelisation (r-NN) r={VOXEL_RADIUS}m")
    # Using a certain radius
    voxelizer = RNN_Voxelisation(df, radius=VOXEL_RADIUS)

    print(f"Animation enabled: {VISUALIZE_ANIMATION}")
    
    # Run voxelisation
    # Generate super-voxels for all points
    s_voxels = voxelizer.voxelize(visualize=VISUALIZE_ANIMATION) 
    
    print("\nVoxelisation Results:")
    print(f"Original Points: {len(df):,}")
    print(f"Super-Voxels:    {len(s_voxels):,}")
    print("\nSample s-voxels:")
    print(s_voxels.head())

    # Visualize Voxelized Data
    # Visualize Voxelized Data
    visualize_point_cloud(s_voxels, window_name=f"Super-Voxels (Averaged Data) ({len(s_voxels):,} voxels)", viz_settings=viz_config)

    # Save Intermediate: Super-Voxel Data
    if SAVE_INTERMEDIATES:
        save_dataframe(s_voxels, f"super_voxels_{len(s_voxels)}.csv", OUTPUT_DIR, None)
        save_point_cloud_ply(s_voxels, f"super_voxels_{len(s_voxels)}.ply", OUTPUT_DIR, None)

    print("\nAll processing complete. Exiting.")

if __name__ == "__main__":
    main()

