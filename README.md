# Segmentation Based Classification of 3D Urban Point Clouds

This project implements the methodology described in the research paper *"Segmentation Based Classification of 3D Urban Point Clouds."* The goal is to semantically classify 3D urban data by processing raw point clouds into, classified objects.

## Phase 1: Data Acquisition

The **Semantic3D** benchmark is utilized as the data source, specifically the `domfountain_station2` dataset.

The input consists of raw 3D point cloud data that contains spatial coordinates ($x, y, z$), RGB color values, and reflectance intensity. Handling this data presents a challenge due to its scale; the dataset comprises over **40 million** individual points. To manage this effectively, the loading phase has been optimized using C-engine parsing, which allows for efficient parsing of the large-scale ASCII format.

## Phase 2: Voxelisation (r-NN)

To address the issue of data redundancy and to provide structure to the sparse point cloud, the **Radius Nearest Neighbor** voxelisation strategy is applied.

During this phase, a fixed radius of $r = 0.1$ **m** is strictly enforced. The output of this process is the grouping of raw points into higher-level structures known as **Super-Voxels**.

## Phase 3: Super-Voxel Transformation

Once the s-voxels are established, each one is transformed into a feature-rich entity by computing a specific set of attributes:

* The **Centroid** $(V_x, V_y, V_z)$, is calculated for each voxel.

* The **Color Properties** are analyzed by determining the mean RGB and Intensity values, along with their respective variances.

* The **Surface Orientation** is determined by computing normal vectors $(V_{nx}, V_{ny}, V_{nz})$ via **Principal Component Analysis**.

# Implementation

This section details the theoretical framework, the mathematical models applied, and the code structure used to execute the project.

## 1. Overview

The processing pipeline operates through five sequential stages:

1. **Data Loading**: The process begins by parsing the raw $(x, y, z, R, G, B, I)$ data from the dataset.

2. **Voxelisation**: Next, the raw data is structured into s-voxels using the r-NN method.

3. **Transformation**: Feature descriptors, such as moments and surface normals, are then computed for each voxel.

4. **Segmentation**: In this stage, the s-voxels are clustered into distinct objects using the **Link-Chain Method**, which relies on both spatial and attribute adjacency.

5. **Classification**: Finally, the resulting segments (e.g., Ground, Building, Tree, Pole, Car) are labeled based on defined geometric rule sets.

## 2. Theoretical Framework

The core mathematical models driving the implementation are defined below.

### 2.1 r-NN Voxelisation

The **r-NN** method partitions the space adaptively using a bottom-up approach. This means voxels are created only where data actually exists, avoiding empty processing.



* **Definition**: A voxel is defined as the set $N$ of points neighboring a random seed $P_{seed}$:

  $$
  N = \{ P_k \in \mathcal{S} \mid \| P_k - P_{seed} \|_2 \le r \}
  $$

  Where:
  *   $N$: The subset of neighboring points forming the voxel.
  *   $P_k$: A candidate point from the global point cloud.
  *   $\mathcal{S}$: The set of all unvisited points in the cloud.
  *   $P_{seed}$: The randomly selected seed point for the voxel.
  *   $r$: The fixed voxelisation radius (0.1 m).

* **Strict Partitioning**: To ensure exclusive assignment, points assigned to set $N$ are immediately removed from the global set $\mathcal{S}$.

### 2.2 Spatial Indexing

To execute the neighborhood search efficiently, a **K-Dimensional Tree** is utilized.

* **Algorithm**: This method recursively bisects the space using axis-aligned hyperplanes (median splitting).

* **Complexity**:

  * Construction: $\mathcal{O}(N \log N)$

  * Query: Average case $\mathcal{O}(\log N)$
  
*   **Visualization**:
    ![KD-Tree Structure](assets/kd_tree_diagram.png)
    *Conceptual visualization of space partitioning using a KD-Tree in a 2D space with x and y coordinates.*

### 2.3 Surface Normal Estimation

Surface normals are estimated via **Principal Component Analysis** of the voxel's covariance matrix $\mathbf{C}$.

1. **Covariance**:

   $$
   \mathbf{C} = \frac{1}{m-1} \sum (\mathbf{x}_k - \mathbf{\mu}) (\mathbf{x}_k - \mathbf{\mu})^T
   $$

   Where:
   *   $\mathbf{C}$: The $3 \times 3$ covariance matrix representing spatial spread.
   *   $m$: The total number of points within the current voxel.
   *   $\mathbf{x}_k$: The spatial coordinates vector $(x, y, z)$ of the $k$-th point.
   *   $\mathbf{\mu}$: The mean position vector (centroid) of the voxel.

2. **Eigen Decomposition**:

   $$
   \mathbf{C} \mathbf{v} = \lambda \mathbf{v}
   $$

   Where:
   *   $\mathbf{v}$: An eigenvector of the covariance matrix.
   *   $\lambda$: The corresponding scalar eigenvalue.

3. **Normal Vector**: The eigenvector $\mathbf{v}_{min}$ corresponding to the **smallest eigenvalue** $\lambda_{min}$ represents the surface normal $\vec{n}$.

### 2.4 Link-Chain Segmentation

The segmentation phase clusters s-voxels into objects using a **Link-Chain** strategy.

* **Linkage Criteria**: Two s-voxels $V_i$ and $V_j$ are linked if they satisfy the following thresholds:

  1. **Spatial Proximity**: $$\| V_{i(xyz)} - V_{j(xyz)} \| < D_{th}$$

  2. **Color Similarity**: $$\| V_{i(RGB)} - V_{j(RGB)} \| < C_{th}$$

  3. **Intensity Similarity**: $$| V_{i(I)} - V_{j(I)} | < I_{th}$$

  Where:
  *   $V_i, V_j$: The two adjacent super-voxels being compared.
  *   $D_{th}$: Threshold distance for spatial proximity.
  *   $C_{th}$: Threshold for RGB color difference.
  *   $I_{th}$: Threshold for intensity difference. This is 1/4 of total intensity range.

* **Chaining**: The transitive closure of these links forms a segment (Connected Component).

### 2.5 Geometric Classification

Finally, segments will be classified based on geometric descriptors derived from the s-voxels.

* **Ground**: These are identified by large planar segments with normal vectors parallel to the global Z-axis ($\vec{n} \approx [0,0,1]$).

* **Buildings**: These segments are characterized by vertical surfaces ($\vec{n} \cdot \vec{z} \approx 0$).

* **Poles**: These are objects exhibiting high vertical linearity ($\lambda_1 \gg \lambda_2 \approx \lambda_3$).

## 3. Code Implementation

The practical execution of the project is handled by specific functional assets implemented in `code/utils.py`. This section details the algorithmic process of each component and how they interact to achieve the theoretical goals.

### 3.1 Data Management

#### `def load_point_cloud(file_path)`

*   **Role**: Serves as the raw data entry point, bridging the file system and the application's memory.
*   **Process**:
    1.  **Validation**: First checks if the target path exists using `os.path.exists` to prevent runtime crashes.
    2.  **Efficient Parsing**: Utilizes `pandas.read_csv` with `engine='c'`. The C-engine is significantly faster than the default Python engine for parsing the 41 million lines of space-separated ASCII data found in the `Semantic3D` dataset.
    3.  **Memory Management**:
        *   Spatial coordinates ($x, y, z$) are strictly cast to `np.float32` (single precision) rather than double precision.
        *   Color/Intensity values ($R, G, B, I$) are cast to `np.uint8` or `np.int32`.
        *   *Why?* This creates a rigid schema that fits the massive dataset into RAM.
*   **Goal**: Returns the standardized `DataFrame` structure required by the `RNN_Voxelisation` class.

### 3.2 Core Logic: `class RNN_Voxelisation`

This class encapsulates the state and behavior required for the Radius Nearest Neighbor method.

#### `def __init__(self, df, radius=0.1)`

*   **Role**: Component Initializer.
*   **Process**:
    1.  **State Retention**: Stores the full pandas DataFrame reference (`self.df`) and the query radius (`self.radius`).
    2.  **Geometry Extraction**: Extracts just the spatial columns into a numpy array: `self.points = df[['x', 'y', 'z']].values`.
*   **Goal**: Prepares the raw coordinate data ($N \times 3$ matrix) specifically for the spatial indexing step.

#### `def build_tree(self)`

*   **Role**: Spatial Indexing Construction.
*   **Process**:
    1.  **KD-Tree Generation**: Wraps `scipy.spatial.cKDTree(self.points)`.
    2.  **Algorithmic Concept**: It recursively partitions the 3D space using axis-aligned hyperplanes to organize points.
*   **Goal**: This asset is the prerequisite for the `voxelize` method. It transforms the neighborhood search problem from a linear scan $\mathcal{O}(N)$ to a logarithmic tree traversal $\mathcal{O}(\log N)$.

#### `def voxelize(self, sample_size=None, random_seed=42)`

*   **Role**: The core processing loop that executes the "Strict Partitioning" set logic.
*   **Process**:
    1.  **State Initialization**:
        *   Creates a boolean array `visited` initialized to `False` for all $N$ points. This represents the global set $\mathcal{S}$.
    2.  **Stochastic Ordering**:
        *   `np.random.permutation` is used to create a shuffled index list.
        *   *Why?* The dataset often stores points in scan-line order. Processing sequentially would create "tube-like" voxels. Random sampling approximates a true Poisson-disk sampling, ensuring voxels grow naturally from unbiased seeds.
    3.  **The Extraction Loop**:
        *   **Seed Selection**: Helper picks the next unvisited index $i$ from the random list.
        *   **Range Query**: Calls `self.tree.query_ball_point(seed, r)`. This utilizes the KD-Tree asset to instantly retrieve all indices $k$ where $\| P_k - P_{seed} \| \le r$.
        *   **Exclusive Assignment**:
            *   Filters indices: `valid = [k for k in neighbors if not visited[k]]`.
            *   Locks points: `visited[valid] = True`. This represents the mathematical operation $\mathcal{S} \leftarrow \mathcal{S} \setminus N$.
    4.  **Feature Computation** (Per Voxel):
        *   **Centroid**: `subset.mean()` computes the geometric center $(V_x, V_y, V_z)$.
        *   **Homogeneity**: `subset.var()` calculates color/intensity variances.
        *   **Surface Normal (PCA)**:
            *   `np.cov`: Computes the $3 \times 3$ covariance matrix of the centered voxel points.
            *   `np.linalg.eigh`: Solves the eigen decomposition.
            *   **Selection**: The eigenvector corresponding to the *smallest* eigenvalue is selected as the normal, as this direction minimizes variance (orthogonal to the surface).
*   **Goal**: Produces the final structured `DataFrame` of Super-Voxels, containing all computed geometric and color features ready for classification.