import os
import sys
from utils import load_point_cloud

def main():
    # Define the dataset path relative to this script
    # Assuming code is in 'code/' and data is in 'data/' (sibling directories)
    # So we go up one level then into data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "domfountain_station2_xyz_intensity_rgb.txt")

    print("--- 3D Urban Point Cloud Segmentation Project ---")
    print("Phase 1: Data Verification")

    # Load data
    df = load_point_cloud(file_path)

    if df is not None:
        print("\nDataset Statistics:")
        print(f"Total Points: {len(df):,}")
        
        print("\nBounding Box:")
        print(f"X: {df['x'].min():.3f} to {df['x'].max():.3f}")
        print(f"Y: {df['y'].min():.3f} to {df['y'].max():.3f}")
        print(f"Z: {df['z'].min():.3f} to {df['z'].max():.3f}")
        
        print("\nColor and Intensity Stats:")
        print(f"Intensity Range: {df['intensity'].min()} - {df['intensity'].max()}")
        print(f"Red Range:       {df['r'].min()} - {df['r'].max()}")

        print("\nSample Data (First 5 rows):")
        print(df.head())

if __name__ == "__main__":
    main()
