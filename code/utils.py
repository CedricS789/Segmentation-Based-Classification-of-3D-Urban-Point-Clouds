import pandas as pd
import numpy as np
import os
import time
from scipy.spatial import cKDTree

def load_point_cloud(file_path):
    """
    Loads a Semantic3D point cloud file (.txt).
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    print(f"Loading {file_path}...")
    start_time = time.time()
    
    try:
        # Semantic3D format: X Y Z Intensity R G B (space separated)
        df = pd.read_csv(
            file_path, 
            sep=' ', 
            header=None, 
            names=['x', 'y', 'z', 'intensity', 'r', 'g', 'b'],
            dtype={
                'x': np.float32, 
                'y': np.float32, 
                'z': np.float32, 
                'intensity': np.int32,
                'r': np.uint8, 
                'g': np.uint8, 
                'b': np.uint8
            },
            engine='c'
        )
        
        elapsed = time.time() - start_time
        print(f"Successfully loaded {len(df):,} points in {elapsed:.2f} seconds.")
        return df

    except Exception as e:
        print(f"Failed to load point cloud: {e}")
        return None

class RNN_Voxelisation:
    """
    Implements the Radius Nearest Neighbor (r-NN) Voxelisation method.
    As described in "Segmentation Based Classification of 3D Urban Point Clouds".
    """
    def __init__(self, df, radius=0.1):
        """
        Args:
            df (pd.DataFrame): Input point cloud with columns ['x', 'y', 'z', 'intensity', 'r', 'g', 'b'].
            radius (float): Radius 'r' for the neighborhood search (in meters).
        """
        self.df = df
        self.radius = radius
        # Extract coordinates for KDTree
        self.points = df[['x', 'y', 'z']].values
        self.tree = None

    def build_tree(self):
        print("Building KDTree...")
        t0 = time.time()
        self.tree = cKDTree(self.points)
        print(f"KDTree built in {time.time() - t0:.2f}s")

    def voxelize(self, sample_size=None, random_seed=42):
        """
        Performs iterative r-NN voxelisation.
        
        Args:
            sample_size (int): If provided, limits the number of voxels generated.
            random_seed (int): Seed for shuffling processing order.
            
        Returns:
            pd.DataFrame: DataFrame of Super-Voxels.
        """
        if self.tree is None:
            self.build_tree()

        n_points = len(self.points)
        visited = np.zeros(n_points, dtype=bool)
        voxels = []

        # Process points in random order to avoid bias
        np.random.seed(random_seed)
        processing_order = np.random.permutation(n_points)
        
        print(f"Starting r-NN Voxelisation (r={self.radius}m)...")
        start_time = time.time()
        
        count = 0
        for i in processing_order:
            if visited[i]:
                continue
            
            # 1. Pick seed point
            seed_point = self.points[i]
            
            # 2. Find neighbors within radius r
            neighbor_indices = self.tree.query_ball_point(seed_point, self.radius)
            
            # Filter out already visited neighbors
            valid_neighbors = [idx for idx in neighbor_indices if not visited[idx]]
            
            if not valid_neighbors:
                continue

            # Mark as visited
            visited[valid_neighbors] = True
            
            # Compute Super-Voxel Attributes
            subset = self.df.iloc[valid_neighbors]
            
            # Mean & Variance (Color/Intensity/XYZ)
            means = subset.mean()
            variances = subset.var()
            
            # Surface Normal (PCA)
            # Covariance matrix of XYZ coordinates
            xyz_subset = subset[['x', 'y', 'z']].values
            if len(xyz_subset) >= 3:
                # Center the points
                centered = xyz_subset - means[['x', 'y', 'z']].values
                cov_matrix = np.cov(centered, rowvar=False)
                # Eigenvalues/vectors
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                # Normal is eigenvector corresponding to smallest eigenvalue
                normal = eigenvectors[:, 0]
            else:
                normal = np.array([0, 0, 1]) # Default vertical normal for degenerate voxels

            # Store result
            # Using notation from the article: V_x, V_y, V_z, V_I, V_r, V_g, V_b
            voxel_data = {
                'V_x': means['x'],
                'V_y': means['y'],
                'V_z': means['z'],
                'V_I': means['intensity'], # Mean Intensity
                'V_r': means['r'],
                'V_g': means['g'],
                'V_b': means['b'],
                'V_var_I': variances['intensity'],
                'V_var_r': variances['r'],
                'V_var_g': variances['g'],
                'V_var_b': variances['b'],
                'V_nx': normal[0],
                'V_ny': normal[1],
                'V_nz': normal[2]
            }

            voxels.append(voxel_data)
            
            count += 1
            if count % 1000 == 0:
                print(f"Created {count} s-voxels... (Visited {np.sum(visited)}/{n_points} points)", end='\r')
            
            if sample_size and count >= sample_size:
                print("\nReached sample limit.")
                break

        print(f"\nVoxelisation complete. Created {len(voxels)} s-voxels in {time.time() - start_time:.2f}s.")
        
        # Convert list of Series to DataFrame
        return pd.DataFrame(voxels)
