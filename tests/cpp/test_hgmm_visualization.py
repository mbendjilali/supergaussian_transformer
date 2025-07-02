# Standard library imports
import math
import os.path as osp
import sys
import time
from pathlib import Path

# Third-party imports
import laspy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gaussian_mixture_cpp import GMMVariant, hierarchical_gmm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Add necessary paths for imports
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(
    osp.dirname(current_dir)
)  # Go up 3 levels to reach project root
sys.path.append(project_root)  # Add project root to Python path

# Add dependencies paths
dependencies_folder = osp.join(project_root, "src/dependencies")
sys.path.append(osp.join(dependencies_folder, "grid_graph/python/bin"))
sys.path.append(osp.join(dependencies_folder, "parallel_cut_pursuit/python/wrappers"))

# Local imports after path setup
from src.data import Data
from src.transforms.point import GroundElevation, PointFeatures
from src.utils.neighbors import knn_1_graph


def make_hierarchy(cluster_size, depth):
    """
    Generate a hierarchy of integers whose product approximates the target cluster size.

    Args:
        cluster_size (int): Target cluster size to approximate
        depth (int): Number of levels in the hierarchy

    Returns:
        list[int]: List of integers in descending order whose product is the nearest
                   power of 2 less than or equal to cluster_size
    """
    # Find the nearest power of 2 less than or equal to cluster_size
    target = 2 ** (cluster_size.bit_length() - 1)
    if target > cluster_size:
        target >>= 1

    # Find the nth root rounded down to nearest power of 2
    base = 2 ** ((target.bit_length() - 1) // depth)

    # Initialize result with base values
    result = [base] * depth

    # Distribute remaining factors (always powers of 2) to maximize first elements
    remaining_power = (target // (base**depth)).bit_length() - 1
    for i in range(remaining_power):
        idx = i % depth
        result[idx] *= 2

    return result


def create_gaussian_ellipsoid(mu, sigma, n_points=12):
    """
    Create ellipsoid points for a 3D Gaussian with minimal point count.
    
    Args:
        mu: Mean vector (3,)
        sigma: Covariance matrix (3, 3)
        n_points: Number of points to generate on the ellipsoid surface
        
    Returns:
        tuple: (x, y, z) coordinates of ellipsoid points
    """
    # Generate points on unit sphere with minimal points
    phi = np.linspace(0, 2*np.pi, n_points)
    theta = np.linspace(-np.pi/2, np.pi/2, n_points)
    phi, theta = np.meshgrid(phi, theta)
    
    # Convert to Cartesian coordinates on unit sphere
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    
    # Stack coordinates
    sphere_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    
    # Compute Cholesky decomposition of covariance matrix
    try:
        L = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        # If not positive definite, use eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(sigma)
        eigenvals = np.maximum(eigenvals, 1e-6)  # Ensure positive
        L = eigenvecs @ np.diag(np.sqrt(eigenvals))
    
    # Transform sphere to ellipsoid
    ellipsoid_points = sphere_points @ L.T + mu
    
    # Reshape back to grid
    x = ellipsoid_points[:, 0].reshape(phi.shape)
    y = ellipsoid_points[:, 1].reshape(phi.shape)
    z = ellipsoid_points[:, 2].reshape(phi.shape)
    
    return x, y, z


def visualize_hgmm_results(x, y, cluster, mu, sigma, output_path, filename, sample_rate):
    """
    Create ultra-lightweight HTML visualization of hGMM results.
    
    Args:
        x: Point cloud coordinates (N, 3)
        y: Ground truth labels (N,)
        cluster: List of cluster assignments for each level
        mu: List of means for each level
        sigma: List of covariances for each level
        output_path: Path to save HTML file
        filename: Original filename for title
        sample_rate: Sample every Nth point for visualization
    """
    num_levels = len(cluster)
    
    # Create subplots - one for each level
    fig = make_subplots(
        rows=1, cols=num_levels,
        subplot_titles=[f'Level {i} ({cluster[i].unique().numel()} clusters)' for i in range(num_levels)],
        specs=[[{'type': 'scatter3d'} for _ in range(num_levels)]]
    )
    
    # Simple color palette
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for level in range(num_levels):
        # Get cluster assignments for this level
        level_clusters = cluster[level]
        unique_clusters = level_clusters.unique()
        
        # Sample points to reduce file size (take every 20th point)
        sample_indices = torch.arange(0, x.shape[0], sample_rate)
        x_sampled = x[sample_indices]
        level_clusters_sampled = level_clusters[sample_indices]
        
        # Add point cloud colored by clusters
        for i, cluster_id in enumerate(unique_clusters):
            mask = level_clusters_sampled == cluster_id
            if mask.sum() == 0:
                continue
                
            color = colors[i % len(colors)]
            
            # Add points for this cluster
            fig.add_trace(
                go.Scatter3d(
                    x=x_sampled[mask, 0].cpu().numpy(),
                    y=x_sampled[mask, 1].cpu().numpy(),
                    z=x_sampled[mask, 2].cpu().numpy(),
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=color,
                        opacity=0.7
                    ),
                    name=f'Cluster {cluster_id}',
                    showlegend=False,
                    hoverinfo='skip'  # Disable hover to reduce file size
                ),
                row=1, col=level + 1
            )
        
        # Add only a few Gaussian ellipsoids to reduce file size
        level_mu = mu[level]
        level_sigma = sigma[level]
        
        # Limit number of Gaussians to display
        max_gaussians = min(3, level_mu.shape[0])
        
        for i in range(max_gaussians):
            # Create ellipsoid for this Gaussian
            x_ell, y_ell, z_ell = create_gaussian_ellipsoid(
                level_mu[i].cpu().numpy(),
                level_sigma[i].cpu().numpy(),
                n_points=10  # Minimal point count
            )
            
            # Add ellipsoid surface
            fig.add_trace(
                go.Surface(
                    x=x_ell,
                    y=y_ell,
                    z=z_ell,
                    colorscale='Viridis',
                    opacity=0.3,
                    showscale=False,
                    name=f'Gaussian {i}',
                    showlegend=False,
                    hoverinfo='skip'  # Disable hover to reduce file size
                ),
                row=1, col=level + 1
            )
    
    # Update layout with minimal settings
    fig.update_layout(
        title=f'hGMM - {Path(filename).name}',
        width=200 * num_levels,
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Update each subplot with minimal settings
    for i in range(num_levels):
        fig.update_scenes(
            dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            row=1, col=i + 1
        )
    
    # Save to HTML with minimal settings
    fig.write_html(
        output_path,
        include_plotlyjs='cdn',  # Use CDN instead of embedding
        full_html=True,
        config={'displayModeBar': False}  # Hide mode bar to reduce size
    )
    print(f"Visualization saved to: {output_path}")


def argument_parser():
    """Create and return an argument parser for the hGMM visualization script."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Hierarchical GMM results")

    # Input/output arguments
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input LAS file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for HTML visualization file",
    )

    # Hierarchical GMM parameters
    parser.add_argument(
        "--cluster_size",
        default=192,
        type=int,
        help="Approximate cluster size",
    )
    parser.add_argument(
        "--depth",
        default=4,
        type=int,
        help="Depth of the hierarchy"
    )
    parser.add_argument(
        "--alpha",
        default=1e0,
        type=float,
        help="Regularization parameter alpha"
    )
    parser.add_argument(
        "--tol",
        default=1e-2,
        type=float,
        help="Convergence tolerance"
    )
    parser.add_argument(
        "--max_iter",
        default=5,
        type=int,
        help="Maximum number of EM iterations"
    )

    # Feature computation parameters
    parser.add_argument(
        "--features",
        default="pos",
        type=str,
        help="Comma-separated list of features to compute (default: pos only)"
    )
    parser.add_argument(
        "--k_neighbors",
        default=50,
        type=int,
        help="Number of neighbors for kNN graph construction"
    )
    parser.add_argument(
        "--downsample_grid_size",
        default=None,
        type=float,
        help="Grid size for downsampling input points"
    )
    parser.add_argument(
        "--sample_rate",
        default=20,
        type=int,
        help="Sample every Nth point for visualization (default: 20)"
    )

    return parser


def downsample_point_cloud(lasdata, grid_size):
    """Downsample point cloud using voxel grid."""
    print(f"Downsampling point cloud with grid size {grid_size}")
    rgb = (
        torch.tensor(
            (
                np.concatenate([lasdata.red, lasdata.green, lasdata.blue], axis=0)
                / 65535
            ).reshape(3, -1).T,
            dtype=torch.float32,
            device="cpu",
        )
        / 255.0
    )
    xyz = torch.tensor(lasdata.xyz, dtype=torch.float32, device="cpu")
    y = torch.tensor(lasdata.classification.copy(), dtype=torch.long, device="cpu")
    y = F.one_hot(y, num_classes=13)
    min_coords = xyz.min(dim=0)[0]
    v = ((xyz - min_coords) / grid_size).long()

    # Create unique voxel IDs
    voxel_ids = (
        (v[:, 0] + v[:, 1] + v[:, 2])
        * (v[:, 0] + v[:, 1] + v[:, 2] + 1)
        * (v[:, 0] + v[:, 1] + v[:, 2] + 2)
        // 6
        + (v[:, 1] + v[:, 2]) * (v[:, 1] + v[:, 2] + 1) // 2
        + v[:, 2]
    )

    # Get unique voxels and mapping
    unique_ids, inverse_indices = voxel_ids.unique(return_inverse=True)

    # Use scatter_mean for efficient centroid computation
    ones = torch.ones_like(voxel_ids, dtype=torch.float)
    counts = torch.zeros(len(unique_ids), device=xyz.device)
    counts.scatter_add_(0, inverse_indices, ones)

    x_downsampled = torch.zeros((len(unique_ids), 3), device=xyz.device)
    rgb_downsampled = torch.zeros((len(unique_ids), 3), device=xyz.device)
    y_downsampled = torch.zeros((len(unique_ids), 13), dtype=torch.int64, device=xyz.device)
    for dim in range(3):
        x_downsampled[:, dim].scatter_add_(0, inverse_indices, xyz[:, dim])
        rgb_downsampled[:, dim].scatter_add_(0, inverse_indices, rgb[:, dim])
    for i in range(13):
        y_downsampled[:, i].scatter_add_(0, inverse_indices, y[:, i])
    x_downsampled /= counts.unsqueeze(1)
    rgb_downsampled /= counts.unsqueeze(1)
    y_downsampled = y_downsampled.argmax(dim=1)
    return x_downsampled, y_downsampled, rgb_downsampled


def main():
    # Parse command line arguments
    parser = argument_parser()
    args = parser.parse_args()

    # Handle output path
    if args.output is None:
        output_path = Path(args.input).parent / f"hgmm_visualization_{Path(args.input).stem}.html"
    else:
        output_path = args.output

    # Parse features
    features_names = [feature.strip() for feature in args.features.split(",")]

    # Create hierarchy
    hierarchy_k = make_hierarchy(args.cluster_size, args.depth)
    print(f"Generated hierarchy: {hierarchy_k} (target size: {math.prod(hierarchy_k)})")

    # Load point cloud
    print(f"Loading data from {args.input}")
    lasdata = laspy.read(args.input)
    
    if args.downsample_grid_size is not None:
        start_time = time.time()
        x, y, rgb = downsample_point_cloud(lasdata, args.downsample_grid_size)
        duration = time.time() - start_time
        print(f"Downsampled number of points: {x.shape[0]} in {duration:.3f} seconds")
    else:
        x = torch.tensor(lasdata.xyz, dtype=torch.float32, device="cpu")
        y = torch.tensor(lasdata.classification.copy(), dtype=torch.int32, device="cpu")
        rgb = (
            torch.tensor(
                (
                    np.concatenate([lasdata.red, lasdata.green, lasdata.blue], axis=0)
                    / 65535
                ).reshape(3, -1).T,
                dtype=torch.float32,
                device="cpu",
            )
            / 255.0
        )

    # Prepare features
    if features_names == ['pos']:
        features = x
    elif features_names == ['pos', 'rgb']:
        features = torch.cat([x, rgb], dim=1)
    else:
        print("Computing k-nearest neighbors...")
        start_time = time.time()
        # Get the k-nearest neighbors for feature computation
        neighbors, distances = knn_1_graph(
            x,
            k=args.k_neighbors,
            r_max=float("inf"),
            batch=None,
            oversample=True,
            self_is_neighbor=False,
            verbose=True,
            trim=False,
        )
        duration = time.time() - start_time
        print(f"KNN completed in {duration:.3f} seconds")

        # Create initial Data object
        data = Data(
            pos=x,
            rgb=rgb,
            neighbor_index=neighbors[1].view(-1, args.k_neighbors),
            neighbor_distance=distances.view(-1, args.k_neighbors),
        )
        
        print("Computing features...")
        start_time = time.time()
        # Compute geometric features
        to_compute = [feature for feature in features_names if feature not in ['pos', 'rgb']]
        if to_compute:
            point_features = PointFeatures(
                keys=to_compute, k_min=5, k_step=-1, overwrite=True
            )
            data = point_features(data)
        duration = time.time() - start_time
        print(f"Geometric features computed in {duration:.3f} seconds")

        features = torch.cat(
            [getattr(data, feature_name) for feature_name in features_names],
            dim=1,
        )

    print(f"\nData ready for hGMM:")
    print(f" - Number of points: {x.shape[0]}")
    print(f" - Feature dimensions: {features.shape}")
    print(f" - Device: {features.device}")

    # Run hGMM with GEM variant
    print(f"\nRunning hGMM with GEM variant...")
    start_time = time.time()
    cluster, _, mu, sigma = hierarchical_gmm(
        features,
        torch.tensor(hierarchy_k, dtype=torch.long, device=features.device),
        args.alpha,
        args.tol,
        args.max_iter,
        GMMVariant.GEM
    )
    duration = time.time() - start_time
    print(f"hGMM completed in {duration:.3f} seconds")

    # Print cluster statistics
    for i, level_clusters in enumerate(cluster):
        num_clusters = level_clusters.unique().numel()
        print(f"Level {i}: {num_clusters} clusters")

    # Create visualization
    print(f"\nCreating lightweight visualization...")
    visualize_hgmm_results(
        x, y, cluster, mu, sigma, output_path, args.input, args.sample_rate
    )

    print(f"\nVisualization complete!")
    print(f"HTML file saved to: {output_path}")
    print(f"Open the HTML file in a web browser to view the interactive 3D visualization.")


if __name__ == "__main__":
    main() 