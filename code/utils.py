import pandas as pd
import numpy as np
import os
import time
from scipy.spatial import cKDTree
import open3d as o3d

def load_point_cloud(file_path, nrows=None):
    """
    Loads a Semantic3D point cloud file (.txt).
    
    Args:
        file_path (str): Path to the point cloud file.
        nrows (int): Number of rows to load. Defaults to None (loads all rows).
    
    Returns:
        pd.DataFrame: DataFrame containing the point cloud data.
    """
    if not os.path.exists(file_path): #Check if file exists
        print(f"Error: File not found at {file_path}") 
        return None

    print(f"Loading {file_path}...")
    start_time = time.time() #Start timer
    
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
            engine='c', #Use C engine for faster parsing
            nrows=nrows
        )
        
        elapsed = time.time() - start_time
        print(f"Successfully loaded {len(df):,} points in {elapsed:.2f} seconds.")
        return df #Return the DataFrame

    except Exception as e: #Catch any exceptions
        print(f"Failed to load point cloud: {e}")
        return None

class RNN_Voxelisation:
    """
    Implements the Radius Nearest Neighbor (r-NN) Voxelisation method.
    As described in "Segmentation Based Classification of 3D Urban Point Clouds" article.
    """
    def __init__(self, df, radius=0.1):
        """
        Initializes the voxelizer.

        Args:
            df (pd.DataFrame): Input point cloud with columns ['x', 'y', 'z', 'intensity', 'r', 'g', 'b'].
            radius (float): Radius 'r' for the neighborhood search (in meters).
        """
        #Defining the Attributes of the class
        self.df = df                                    #Input point cloud
        self.radius = radius                            #Radius 'r' for the neighborhood search (in meters)
        self.points_cloud = df[['x', 'y', 'z']].values  #Extract coordinates of the points cloud (the actual 3D points)
        self.tree = None                                #KDTree


    def build_tree(self):
        """Builds the KDTree for the point cloud."""
        print("Building KDTree...")
        t0 = time.time() #Start timer
        self.tree = cKDTree(self.points_cloud)          #Build KDTree using the scipy library. cKDTree is a C implementation of the KDTree and is much faster than the Python implementation.
        print(f"KDTree built in {time.time() - t0:.2f}s")


    def voxelize(self, sample_size=None, random_seed=42, visualize=False):
        """
        Performs iterative r-NN voxelisation.
        
        Args:
            sample_size (int): If provided, limits the number of voxels generated.
            random_seed (int): Seed for shuffling processing order.
            visualize (bool): If True, animates the process in a window.
            
        Returns:
            pd.DataFrame: DataFrame of Super-Voxels.
        """
        #If the tree is not built, build it
        if self.tree is None:
            self.build_tree()

        n_points = len(self.points_cloud)                       #Number of points
        is_index_visited = np.zeros(n_points, dtype=bool)       #Array of boolean values telling if an index has been visited or not (default to False)
        super_voxels = []                                       #Initialize super_voxels as an empty list

        # Process points in random order to avoid bias
        np.random.seed(random_seed)                             #Set the random seed index such that the results are reproducible
        shuffled_indices = np.random.permutation(n_points)      #This represents the order in which the points will be processed (Randomized)
        
        print(f"Starting r-NN Voxelisation (r={self.radius}m)...")
        start_time = time.time()                                #Start timer
        
        # Visualization Setup
        vis = None              #Initialize the visualizer
        main_pcd = None         #Initialize the main point cloud
        central_sphere = None   #Initialize the root sphere
        sv_sphere = None        #Initialize the super voxel sphere
        
        if visualize:
            print("Initializing 3D Animation...")
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="r-NN Voxelization Process", width=1024, height=768)
            
            # Create main point cloud
            main_pcd = o3d.geometry.PointCloud()
            main_pcd.points = o3d.utility.Vector3dVector(self.points_cloud)

            # Default color: Grey
            colors = np.ones((n_points, 3)) * 0.7 
            main_pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(main_pcd)
            
            # Placeholders for Seed and SuperVoxel
            central_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius/5)
            central_sphere.paint_uniform_color([1, 0, 0]) # Red
            vis.add_geometry(central_sphere)
            
            # We can't easily add/remove geometries rapidly without overhead, 
            # so we just modify the main cloud colors and move the markers.

        s_voxel_count = 0 #Used to count the number of voxels
        try:
            for current_point_index in shuffled_indices:
                if is_index_visited[current_point_index]:
                    continue
                
                # 1. Pick root point
                centroid_point = self.points_cloud[current_point_index]
                
                # 2. Find neighbors of root_point within radius r using a scipy method
                neighbor_points_indices = self.tree.query_ball_point(centroid_point, self.radius) 
                
                # Filter out already visited neighbors
                valid_neighbor_points_indices = [idx for idx in neighbor_points_indices if not is_index_visited[idx]]
                
                if not valid_neighbor_points_indices:
                    continue

                # Visualization Update (Before processing)
                if visualize:
                    # Update Seed Marker
                    central_sphere.translate(centroid_point - central_sphere.get_center(), relative=True)
                    vis.update_geometry(central_sphere)
                    
                    # Highlight Neighbors (Green)
                    np_colors = np.asarray(main_pcd.colors)
                    np_colors[valid_neighbor_points_indices] = [0, 1, 0]

                    # Highlight Seed (Red) - redundant with sphere but good for clarity
                    np_colors[current_point_index] = [1, 0, 0]
                    
                    main_pcd.colors = o3d.utility.Vector3dVector(np_colors)
                    vis.update_geometry(main_pcd)
                    
                    vis.poll_events()
                    vis.update_renderer()
                    
                    # specific delay
                    # time.sleep(0.01) 

                # Mark as visited
                is_index_visited[valid_neighbor_points_indices] = True
                
                # Compute Super-Voxel Attributes
                subset = self.df.iloc[valid_neighbor_points_indices]        #(iloc = Integer Location). Returns the indices of the valid neighbors to the current centroid point and puts them in a subset together
                
                # Mean & Variance (Color/Intensity/XYZ)
                means = subset.mean()                                       #Computes the mean of the subset for each point attribute (x, y, z, intensity, r, g, b)
                variances = subset.var()                                    #Computes the variance of the subset for each point attribute (x, y, z, intensity, r, g, b)
                
                # Surface Normal (Principal Component Analysis)
                # Covariance matrix of XYZ coordinates
                xyz_subset = subset[['x', 'y', 'z']].values                 #Extracts the x, y, z coordinates of the subset
                
                if len(xyz_subset) >= 3:
                    
                    # Center the points
                    centered = xyz_subset - means[['x', 'y', 'z']].values   #Centers the points arround the origin
                    cov_matrix = np.cov(centered, rowvar=False)

                    # Eigenvalues/vectors
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  #Computes the eigenvalues and eigenvectors of the covariance matrix

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

                super_voxels.append(voxel_data)
                
                # Visualization Update (After processing)
                if visualize:
                    
                    # Mark processed points as Dark Grey (done)
                    np_colors = np.asarray(main_pcd.colors)
                    np_colors[valid_neighbor_points_indices] = [0.2, 0.2, 0.2]
                    main_pcd.colors = o3d.utility.Vector3dVector(np_colors)
                    vis.update_geometry(main_pcd)
                    vis.poll_events()
                    vis.update_renderer()

                s_voxel_count += 1
                if s_voxel_count % 1000 == 0:
                    print(f"Created {s_voxel_count} s-voxels... (Visited {np.sum(is_index_visited)}/{n_points} points)", end='\r')
                
                if sample_size and s_voxel_count >= sample_size:
                    print("\nReached sample limit.")
                    break
        finally:
            if visualize:
                vis.destroy_window()

        print(f"\nVoxelisation complete. Created {len(super_voxels)} s-voxels in {time.time() - start_time:.2f}s.")
        
        # Convert list of Series to DataFrame
        return pd.DataFrame(super_voxels)


def visualize_point_cloud(df, window_name="Point Cloud"):
    """
    Helper function to visualize the point cloud using Open3D.
    Handles both raw data (x, y, z, r, g, b) and voxel data (V_x, V_y, V_z, V_r, V_g, V_b).
    
    Args:
        df (pd.DataFrame): The dataframe containing point cloud data.
        window_name (str): Title of the visualization window.
    """
    print(f"Preparing visualization for: {window_name}...")
    
    # Check for column names
    if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
        points = df[['x', 'y', 'z']].values
        if 'r' in df.columns and 'g' in df.columns and 'b' in df.columns:
            colors = df[['r', 'g', 'b']].values / 255.0 # Normalize to [0, 1]
        else:
            colors = None
    elif 'V_x' in df.columns and 'V_y' in df.columns and 'V_z' in df.columns:
        points = df[['V_x', 'V_y', 'V_z']].values
        if 'V_r' in df.columns and 'V_g' in df.columns and 'V_b' in df.columns:
            colors = df[['V_r', 'V_g', 'V_b']].values / 255.0
        else:
            colors = None
    else:
        print("Error: DataFrame does not contain recognized coordinate columns (x,y,z or V_x,V_y,V_z).")
        return

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    print(f"Opening 3D window for '{window_name}'. Close the window to continue...")
    o3d.visualization.draw_geometries([pcd], window_name=window_name)
    print("Window closed.")

