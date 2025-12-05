import pandas as pd
import numpy as np
import os
import time

def load_point_cloud(file_path):
    """
    Loads a Semantic3D point cloud file (.txt).
    
    Args:
        file_path (str): Path to the .txt file.
        
    Returns:
        pd.DataFrame: DataFrame containing the point cloud data with columns:
                      ['x', 'y', 'z', 'intensity', 'r', 'g', 'b']
        None: If the file does not exist or fails to load.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    print(f"Loading {file_path}...")
    start_time = time.time()
    
    try:
        # Semantic3D format: X Y Z Intensity R G B (space separated)
        # Using pandas for speed on large CSV/TXT files
        # Header is None because these are raw ASCII files without a header row
        df = pd.read_csv(
            file_path, 
            sep=' ', 
            header=None, 
            names=['x', 'y', 'z', 'intensity', 'r', 'g', 'b'],
            dtype={
                'x': np.float32, 
                'y': np.float32, 
                'z': np.float32, 
                'intensity': np.int32, # Intensity is usually integer-like
                'r': np.uint8, 
                'g': np.uint8, 
                'b': np.uint8
            },
            engine='c' # C engine is faster
        )
        
        elapsed = time.time() - start_time
        print(f"Successfully loaded {len(df):,} points in {elapsed:.2f} seconds.")
        return df

    except Exception as e:
        print(f"Failed to load point cloud: {e}")
        return None
