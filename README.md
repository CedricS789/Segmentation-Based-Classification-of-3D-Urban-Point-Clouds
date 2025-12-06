# Project Implementation Plan: Segmentation Based Classification of 3D Urban Point Clouds

## 1. Data Acquisition and Preprocessing

First acquire the point cloud data. The paper typically uses data from the Blaise Pascal University or the 3D Urban Data Challenge.

- **Input**: Raw 3D point cloud data containing spatial coordinates (x, y, z), RGB color, and reflectance intensity.
- **Process**: If the dataset is large, you may need to downsample or filter noise as a preprocessing step before voxelization.

### 2. Voxelization

Reduce the data complexity by grouping raw points into voxels.

- **Method**: Select a center point and find all neighboring points within a fixed radius (r-NN) to form a voxel.
- **Constraint**: The maximum voxel size is predefined (e.g., 0.3m - 0.5m), but actual sizes vary based on the min/max values of the contained points.
- **Optimization**: Once points are assigned to a voxel, remove them from the pool of candidates to prevent over-segmentation.

### 3. Transformation into Super-Voxels

Calculate specific attributes for each voxel to transform it into a "super-voxel" (s-voxel).

- **Geometric Center**: Compute the centroid ($V_{X,Y,Z}$).
- **Color & Intensity**: Calculate the mean and variance for RGB values ($V_{R,G,B}$) and reflectance intensity ($V_I$).
- **Surface Normal**: Estimate the normal vector for the points in the voxel using Principal Component Analysis (PCA).

### 4. Segmentation via Link-Chain Method

Cluster adjacent s-voxels into distinct objects.

- **Linkage Strategy**: Unlike traditional region growing, select any s-voxel as a **principal link** and identify **secondary links** (neighbors).
- **Thresholds**: Link s-voxels only if they satisfy specific conditions regarding spatial distance, color difference, and intensity difference.
- **Chaining**: Connect principal links to form a continuous chain, identifying a complete segmented object.

### 5. Classification

Assign a class label (Building, Road, Pole, Car, Tree) to each segmented object.

- **Ground Removal**: First, identify and segment the ground/road assuming it is a flat plane.
- **Descriptor Analysis**: Classify the remaining "floating" objects by comparing their properties against predefined geometric models and thresholds.
    - **Buildings**: Surface normals predominantly parallel to the ground plane.
    - **Trees**: Height difference between geometrical center and barycenter > 0.
    - **Poles**: Long, thin vertical shapes.
    - **Cars**: Broad and short shapes.

### 6. Evaluation

Implement the specific metrics defined in the paper to verify your results.

- **Confusion Matrix**: Construct a matrix based on voxel counts.
- **Metrics**: Calculate Segmentation Accuracy (SACC), Classification Accuracy (CACC), and their overall averages (OSACC, OCACC).

---

# Radius Nearest Neighbor (r-NN) Voxelization

## 1\. Conceptual Framework

### 1.1 Standard vs. r-NN Voxelization

Standard voxelization techniques typically apply a fixed, rigid grid over the 3D space (top-down approach). This "cookie-cutter" method often results in:

  * **Split objects:** Points belonging to a single structural element may be arbitrarily divided across voxel boundaries.
  * **Inefficient memory usage:** Voxels are created in empty space or contain very few points while occupying full memory.

The r-NN method employs a "shrink-wrapping" strategy (bottom-up approach). Volumetric units are created only where data exists, and their dimensions adapt to tightly fit the local cluster of points.

## 2\. Mathematical Formulation

The algorithm processes a global set of raw 3D points $P$ to generate a set of super-voxels $S$. The process iterates until all points in $P$ have been assigned to a voxel.

### 2.1 Radius Neighborhood Search

A center point $p_c$ is selected from the set of available points $P$. A neighborhood set $N$ is defined by finding all points within a fixed Euclidean distance $r$ (the radius) from $p_c$.

$$N = \{p_i \in P \mid ||p_i - p_c|| \le r\}$$

  * **$p_c$**: The seed point for the current voxel.
  * **$r$**: The predefined maximum radius, which determines the maximum possible extent of a voxel.
  * **$||\cdot||$**: The Euclidean distance metric.

### 2.2 Voxel Formation (Adaptive AABB)

While the search region is spherical, the resulting voxel is defined as an Axis-Aligned Bounding Box (AABB). This cuboid shape is preferred for its symmetry and ease of property extraction.

The dimensions of the voxel ($s_x, s_y, s_z$) are dynamic and determined by the spatial spread of the points in $N$:

$$s_{dim} = \max_{p \in N}(p_{dim}) - \min_{p \in N}(p_{dim})$$

Where $dim \in \{x, y, z\}$.

  * **Structural Adaptability:** The voxel size ensures the profile of the structure is maintained. For planar surfaces (e.g., roads), one dimension will be significantly smaller than the others. For linear structures (e.g., poles), two dimensions will be minimized.
  * **Constraint:** The dimension of the voxel is strictly bounded by the search diameter $2r$.

### 2.3 The Exclusion Principle

To prevent over-segmentation and redundant processing, points assigned to a voxel are removed from the global search pool immediately after assignment.

$$P_{remaining} = P_{current} \setminus N$$

This ensures that every 3D point belongs to exactly one super-voxel, significantly reducing the total dataset size ($s \ll p$, where $s$ is the number of voxels and $p$ is the number of points).

## 3\. Implementation Strategy

To ensure computational efficiency suitable for large point clouds, the following implementation strategies are recommended:

1.  **Spatial Indexing (KD-Tree):**
    Calculating Euclidean distances for all points is an $O(N^2)$ operation. A KD-Tree structure must be used to perform the radius search ($range\_search$) in logarithmic time.

2.  **State Management (Visited Mask):**
    Instead of physically deleting points from arrays (which is computationally expensive), a boolean mask (e.g., `visited[N]`) should be maintained.

      * Initialize `visited` to `False`.
      * Select $p_c$ only if `visited[index] == False`.
      * Update `visited[indices_in_N] = True` after voxel creation.

3.  **Attribute Calculation:**
    Once $N$ is established, the super-voxel attributes (Centroid, Mean Color, Mean Intensity, Variance) are computed immediately before proceeding to the next iteration.


# Functions
This document details the implementation of the **Voxelisation Phase** in `code/utils.py`, establishing a direct correspondence between the Python code and the mathematical theory presented in *"Segmentation Based Classification of 3D Urban Point Clouds"*.

## 1. Nomenclature & Definitions

We adopt the following notation from the article:

-   $\mathcal{S}$: The set of raw 3D points acquired from the LIDAR/Photogrammetry.
-   $P_i = (x_i, y_i, z_i, R_i, G_i, B_i, I_i)$: A single point in $\mathcal{S}$.
-   $V_j$: The $j$-th **Super-Voxel** (s-voxel) generated by the algorithm.
-   $\mathcal{N}_r(P_{seed})$: The neighborhood of a seed point defined by radius $r$.

The output of the voxelisation phase is a set of s-voxels, where each $V_j$ is characterized by:
-   **Geometric Center**: $V_{X,Y,Z}$
-   **Mean Color**: $V_{R,G,B}$
-   **Mean Intensity**: $V_I$
-   **Variances**: $\sigma^2_{R,G,B}, \sigma^2_I$
-   **Surface Normal**: $\vec{n} = (n_x, n_y, n_z)$

---

## 2. The r-NN Voxelisation Algorithm

The class `RNN_Voxelisation` implements the **Radius Nearest Neighbor (r-NN)** partitioning strategy.

### 2.1 Theoretical Definition
The goal is to partition $\mathcal{S}$ into subsets such that each point belongs to exactly one voxel. The process is iterative:
1.  Select a random seed point $P_{seed} \in \mathcal{S}$ (that has not been visited).
2.  Define the voxel $\mathcal{V}$ by finding all points $P_k$ such that:
    $$ \| P_k - P_{seed} \|_2 \le r $$
    $$ \sqrt{(x_k - x_{seed})^2 + (y_k - y_{seed})^2 + (z_k - z_{seed})^2} \le r $$
3.  Compute attributes ($\mu, \sigma^2, \vec{n}$) for $\mathcal{V}$.
4.  Remove these points from $\mathcal{S}$ (mark as `visited`).
5.  Repeat until $\mathcal{S}$ is empty.

### 2.2 Code Implementation
*   **Initialization**: We build a **KDTree** (`scipy.spatial.cKDTree`) on the XYZ coordinates of $\mathcal{S}$. This allows us to solve the range query $\| P_k - P_{seed} \| \le r$ in $\mathcal{O}(\log N)$ time instead of $\mathcal{O}(N)$.
*   **Loop**:
    ```python
    neighbor_indices = self.tree.query_ball_point(seed_point, self.radius)
    ```
    This line directly implements the inequality above. The `radius` parameter corresponds to $r$ in the article (typically $0.1m$).

---

## 3. Transformation: Computing s-voxel Attributes

Once a set of points $\mathcal{P}_{voxel} = \{P_1, \dots, P_m\}$ is identified for a voxel, we compute the descriptors.

### 3.1 Geometric Center & Means ($V_{X,Y,Z}, V_{R,G,B}, V_I$)
The geometric center is the centroid of the points:
$$ V_{X,Y,Z} = \frac{1}{m} \sum_{k=1}^{m} \mathbf{x}_k $$
In code:
```python
means = subset.mean()
# Implements: V_x = mean(x), V_y = mean(y), ...
```

### 3.2 Variances ($V_{var}$)
Variance describes the homogeneity of the voxel (e.g., is the color consistent?).
$$ \sigma^2_I = \frac{1}{m-1} \sum_{k=1}^{m} (I_k - V_I)^2 $$
In code:
```python
variances = subset.var()
```

### 3.3 Surface Normal Estimation (PCA)
The article specifies using **Principal Component Analysis (PCA)** to estimate the local surface orientation $\vec{n}$.

**Theory**:
The surface normal is the direction in which the data varies the *least* (orthogonal to the surface plane). We compute the **Covariance Matrix** $\mathbf{C}$ of the XYZ coordinates:
$$ \mathbf{C} = \frac{1}{m-1} \sum_{k=1}^{m} (\mathbf{x}_k - V_{XYZ}) (\mathbf{x}_k - V_{XYZ})^T $$

We then solve the eigenvalue problem:
$$ \mathbf{C} \mathbf{v}_i = \lambda_i \mathbf{v}_i $$
where $\lambda_1 \ge \lambda_2 \ge \lambda_3$ are the eigenvalues and $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$ are the eigenvectors.

*   $\mathbf{v}_1, \mathbf{v}_2$: Tangent to the surface (directions of max spread).
*   $\mathbf{v}_3$: **Normal to the surface** (direction of min spread).

**Code Mapping**:
```python
# 1. Center the data (x - mean)
centered = xyz_subset - means[['x', 'y', 'z']].values

# 2. Compute Covariance Matrix C
cov_matrix = np.cov(centered, rowvar=False)

# 3. Eigen Decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 4. Select Normal (corresponding to smallest eigenvalue)
# np.linalg.eigh returns sorted eigenvalues, so index 0 is min(lambda)
normal = eigenvectors[:, 0] 
```
This precisely implements the theory: $\vec{n} \approx \mathbf{v}_3$.

## 4. Output Data Structure
The final `pd.DataFrame` contains strictly the columns defined in the project's feature set:
*   `V_x`, `V_y`, `V_z`
*   `V_I`, `V_r`, `V_g`, `V_b`
*   `V_var_I`, `V_var_r`, `V_var_g`, `V_var_b`
*   `V_nx`, `V_ny`, `V_nz`
