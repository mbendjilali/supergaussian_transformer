# Standard library imports
import argparse
import math
import os.path as osp
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any

# Third-party imports
import laspy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gaussian_mixture_cpp import GMMVariant, hierarchical_gmm

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


def add_gaussian_noise(x: torch.Tensor, snr_db: float, noise_level: Optional[float] = None) -> Tuple[torch.Tensor, float]:
    """Add Gaussian noise to point cloud coordinates to achieve desired SNR or noise level.
    
    Args:
        x (torch.Tensor): Input points of shape (N, 3)
        snr_db (float): Signal-to-noise ratio in decibels. Use float('inf') for no noise.
        noise_level (Optional[float]): Direct noise level (variance) to use instead of SNR.
        
    Returns:
        Tuple[torch.Tensor, float]: Tuple containing:
            - Noisy points (or clean points if snr_db is infinite)
            - Noise variance (0.0 if snr_db is infinite)
    """
    # Handle infinite SNR case (no noise)
    if snr_db == float('inf') and noise_level is None:
        return x, 0.0
        
    # If noise_level is provided, use it directly
    if noise_level is not None:
        noise_power = noise_level
    else:
        # Calculate signal power (variance of coordinates)
        signal_power = torch.var(x, dim=0).mean()
        
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)
        
        # Calculate noise power
        noise_power = signal_power / snr_linear
    
    # Generate noise
    noise = torch.randn_like(x) * torch.sqrt(torch.tensor(noise_power / 2, device=x.device))
    
    # Add noise to coordinates
    noisy_x = x + noise
    
    # Calculate actual noise variance
    noise_variance = torch.var(noise, dim=0).mean().item()
    
    return noisy_x, noise_variance


def compute_snr(x_clean: torch.Tensor, x_noisy: torch.Tensor) -> float:
    """Compute SNR between clean and noisy point clouds.
    
    Args:
        x_clean (torch.Tensor): Clean points
        x_noisy (torch.Tensor): Noisy points
        
    Returns:
        float: SNR in decibels
    """
    # Calculate signal power (variance of clean coordinates)
    signal_power = torch.var(x_clean, dim=0).mean()
    
    # Calculate noise power (mean squared error)
    noise_power = torch.mean((x_clean - x_noisy) ** 2)
    
    # Compute SNR in dB
    snr_db = 10 * torch.log10(signal_power / noise_power)
    
    return snr_db.item()


def batch_sqrt_lower_triangular(L: torch.Tensor) -> torch.Tensor:
    """Compute batch-wise square root of lower triangular matrices.
    
    Args:
        L (torch.Tensor): Lower triangular matrices of shape (B, 3, 3)
        
    Returns:
        torch.Tensor: Square root matrices of shape (B, 3, 3)
    """
    S = torch.zeros_like(L)
    S[:, 0, 0] = torch.sqrt(L[:, 0, 0])
    S[:, 1, 0] = L[:, 1, 0] / S[:, 0, 0]
    S[:, 1, 1] = torch.sqrt(L[:, 1, 1] - S[:, 1, 0] ** 2)
    S[:, 2, 0] = L[:, 2, 0] / S[:, 0, 0]
    S[:, 2, 1] = (L[:, 2, 1] - S[:, 1, 0] * S[:, 2, 0]) / S[:, 1, 1]
    S[:, 2, 2] = torch.sqrt(L[:, 2, 2] - S[:, 2, 0] ** 2 - S[:, 2, 1] ** 2)
    return S


def make_hierarchy(approx_cluster_size: int, depth: int) -> List[int]:
    """Generate a hierarchy of integers whose product approximates the target cluster size.
    
    Args:
        approx_cluster_size (int): Target cluster size to approximate
        depth (int): Number of levels in the hierarchy
        
    Returns:
        List[int]: List of integers in descending order whose product is the nearest
                   power of 2 less than or equal to approx_cluster_size
    """
    target = 2 ** (approx_cluster_size.bit_length() - 1)
    if target > approx_cluster_size:
        target >>= 1

    base = 2 ** ((target.bit_length() - 1) // depth)
    result = [base] * depth

    remaining_power = (target // (base**depth)).bit_length() - 1
    for i in range(remaining_power):
        idx = i % depth
        result[idx] *= 2

    return result


def hard_assign(labels: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Vectorized implementation to assign most common y value to each label.
    
    Args:
        labels (torch.Tensor): Cluster assignment for each point (N,)
        y (torch.Tensor): Ground truth labels (N,)
        
    Returns:
        torch.Tensor: Most common y value for each cluster
    """
    labels = labels.long()
    y = y.long()
    
    unique_labels = labels.unique()
    num_classes = y.max().item() + 1
    
    label_to_idx = torch.zeros(unique_labels.max().item() + 1, dtype=torch.long, device=labels.device)
    label_to_idx[unique_labels] = torch.arange(len(unique_labels), device=labels.device)
    
    label_idx = label_to_idx[labels]
    
    votes = torch.zeros((len(unique_labels), num_classes), device=labels.device)
    votes.index_put_((label_idx, y), torch.ones(len(y), device=labels.device), accumulate=True)
    
    most_common_classes = votes.argmax(dim=1)
    result = most_common_classes[label_idx]
    
    return result


def compute_mean_accuracy(predictions: torch.Tensor, y: torch.Tensor) -> float:
    """Compute mean class accuracy for semantic segmentation.
    
    Args:
        predictions (torch.Tensor): Predicted labels (N,)
        y (torch.Tensor): Ground truth labels (N,)
        
    Returns:
        float: Mean class accuracy
    """
    unique_classes = y.unique()
    accuracies = []
    for cls in unique_classes:
        mask = y == cls
        if mask.sum() == 0:
            continue

        class_acc = (predictions[mask] == y[mask]).float().mean()
        accuracies.append(class_acc)

    return torch.stack(accuracies).mean().item()


def compute_miou(predictions: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    """Compute mean IoU for semantic segmentation.
    
    Args:
        predictions (torch.Tensor): Predicted labels (N,)
        y (torch.Tensor): Ground truth labels (N,)
        num_classes (int): Number of semantic classes
        
    Returns:
        float: Mean IoU across all classes
    """
    intersections = torch.zeros(num_classes, device=y.device)
    unions = torch.zeros(num_classes, device=y.device)

    for cls in range(num_classes):
        pred_mask = predictions == cls
        true_mask = y == cls
        intersection = (pred_mask & true_mask).sum().float()
        union = (pred_mask | true_mask).sum().float()

        intersections[cls] = intersection
        unions[cls] = union
    valid_classes = unions > 0
    if valid_classes.sum() == 0:
        return 0.0

    ious = intersections[valid_classes] / unions[valid_classes]
    return ious.mean().item()

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
    y = torch.tensor(lasdata.semantic_label.copy(), dtype=torch.long, device="cpu")
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
    y_downsampled = torch.zeros((len(unique_ids), 8), dtype=torch.int64, device=xyz.device)
    for dim in range(3):
        x_downsampled[:, dim].scatter_add_(0, inverse_indices, xyz[:, dim])
        rgb_downsampled[:, dim].scatter_add_(0, inverse_indices, rgb[:, dim])
    for i in range(8):
        y_downsampled[:, i].scatter_add_(0, inverse_indices, y[:, i])
    x_downsampled /= counts.unsqueeze(1)
    rgb_downsampled /= counts.unsqueeze(1)
    y_downsampled = y_downsampled.argmax(dim=1)
    return x_downsampled, y_downsampled, rgb_downsampled


def test_em(
    x: torch.Tensor,
    y: torch.Tensor,
    hierarchy_k: List[int],
    alpha: float,
    tol: float,
    max_iter: int,
    variant: GMMVariant,
    over_iter: int,
    noise_variance: float = 0.0
) -> Tuple[Dict[str, List[float]], torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Test GMM variants with noisy data.
    
    Args:
        x (torch.Tensor): Input features
        y (torch.Tensor): Ground truth labels
        hierarchy_k (List[int]): Number of clusters for each level
        alpha (float): Regularization parameter
        tol (float): Convergence tolerance
        max_iter (int): Maximum number of iterations
        variant (GMMVariant): GMM variant to use
        over_iter (int): Number of additional iterations after convergence
        noise_variance (float): Variance of the added noise (0.0 for clean data)
        
    Returns:
        Tuple containing:
        - Dict[str, List[float]]: Results dictionary with metrics
        - torch.Tensor: Predictions
        - torch.Tensor: Error mask
        - List[torch.Tensor]: Cluster assignments for each level
        - List[torch.Tensor]: Means for each level
        - List[torch.Tensor]: Covariances for each level
    """
    hierarchy_k = torch.tensor(hierarchy_k, dtype=torch.long, device=x.device)
    results_dict = {
        "duration": [],
        "accuracy": [],
        "miou": [],
        "iterations": [],
        "noise_variance": noise_variance,
    }

    for i in range(over_iter + 1):
        start_time = time.time()
        cluster, _, mu, sigma = hierarchical_gmm(
            x, hierarchy_k, alpha, tol, max_iter + i, variant
        )
        duration = time.time() - start_time
        print(f"Hierarchical GMM completed in {duration:.3f} seconds")

        start_time = time.time()
        predictions = hard_assign(cluster[-1], y)
        error = (predictions != y).int()
        duration = time.time() - start_time
        print(f"Mapping labels back to original points completed in {duration:.3f} seconds")

        start_time = time.time()
        accuracy = compute_mean_accuracy(predictions, y)
        duration = time.time() - start_time
        print(f"Accuracy: {accuracy:.3f} in {duration:.3f} seconds")

        start_time = time.time()
        miou = compute_miou(predictions, y, num_classes=8)
        duration = time.time() - start_time
        print(f"mIoU: {miou:.3f} in {duration:.3f} seconds")

        results_dict["duration"].append(duration)
        results_dict["iterations"].append(max_iter + i)
        results_dict["accuracy"].append(accuracy)
        results_dict["miou"].append(miou)

    return results_dict, predictions, error, cluster, mu, sigma


def baseline(x: torch.Tensor, y: torch.Tensor, grid_size: float, filename: str, noise_variance: float = 0.0) -> Dict[str, Any]:
    """Compute baseline metrics using voxel grid partitioning.
    
    Args:
        x (torch.Tensor): Input points
        y (torch.Tensor): Ground truth labels
        grid_size (float): Size of voxel grid
        filename (str): Name of input file for logging
        noise_variance (float): Variance of the added noise (0.0 for clean data)
        
    Returns:
        Dict[str, Any]: Results dictionary with metrics
    """
    results_dict = {
        "duration": [],
        "accuracy": [],
        "miou": [],
        "iterations": [],
        "point_count": x.shape[0],
        "filename": filename,
        "variant": "GRID",
        "actual_size": 0,
        "depth": 1,
        "noise_variance": noise_variance,
    }
    start_time = time.time()
    
    min_coords = x.min(dim=0)[0]
    v = ((x - min_coords) / grid_size).long()

    labels = (
        (v[:, 0] + v[:, 1] + v[:, 2])
        * (v[:, 0] + v[:, 1] + v[:, 2] + 1)
        * (v[:, 0] + v[:, 1] + v[:, 2] + 2)
        // 6
        + (v[:, 1] + v[:, 2]) * (v[:, 1] + v[:, 2] + 1) // 2
        + v[:, 2]
    )
    predictions = hard_assign(labels, y)
    duration = time.time() - start_time
    
    num_voxels = labels.unique().numel()
    accuracy = compute_mean_accuracy(predictions, y)
    miou = compute_miou(predictions, y, num_classes=8)
    print(f"Number of non-empty voxels: {num_voxels}, Accuracy: {accuracy:.3f}, mIoU: {miou:.3f} in {duration:.3f} seconds")

    results_dict["duration"].append(duration)
    results_dict["accuracy"].append(accuracy)
    results_dict["miou"].append(miou)
    results_dict["iterations"].append(1)
    results_dict["actual_size"] = num_voxels

    return results_dict


def direction_features(pos: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    """Compute direction features for points based on their neighbors.
    
    Args:
        pos (torch.Tensor): Point positions
        neighbors (torch.Tensor): Neighbor indices
        
    Returns:
        torch.Tensor: Direction features
    """
    n_points = pos.shape[0]
    neighbor_pos = pos[neighbors.view(-1)].view(n_points, -1, 3)
    
    centers = pos.unsqueeze(1)
    centered = neighbor_pos - centers
    
    cov = torch.bmm(centered.transpose(1, 2), centered) / (neighbors.shape[1] - 1)
    
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    max_values, max_indices = torch.max(eigenvalues, dim=1)
    directions = max_values[:, None] * eigenvectors[torch.arange(n_points), max_indices]
    
    return directions


def save_noisy_scene(
    x_noisy: torch.Tensor,
    y: torch.Tensor,
    rgb: torch.Tensor,
    snr_db: float,
    noise_level: Optional[float],
    original_file: Path,
    trial: int
) -> None:
    """Save a noisy point cloud scene as a LAS file.
    
    Args:
        x_noisy (torch.Tensor): Noisy point coordinates
        y (torch.Tensor): Semantic labels
        rgb (torch.Tensor): RGB colors
        snr_db (float): SNR level in dB (use float('inf') for clean data)
        noise_level (Optional[float]): Direct noise level (variance) if used
        original_file (Path): Path to original LAS file
        trial (int): Trial number
    """
    # Create output directory if it doesn't exist
    output_dir = original_file.parent / "noisy_scenes"
    output_dir.mkdir(exist_ok=True)
    
    # Create output filename
    if snr_db == float('inf'):
        output_file = output_dir / f"clean_{original_file.name}"
    else:
        # Format SNR as integer if it's a whole number, otherwise use 1 decimal place
        snr_str = int(snr_db)
        output_file = output_dir / f"snr_{snr_str}dB_trial_{trial}_{original_file.name}"
    
    # Create new LAS file with same header as input
    las = laspy.LasData(laspy.LasHeader(version="1.4", point_format=7))
    
    # Add core XYZ coordinates
    las.x = x_noisy[:, 0].numpy()
    las.y = x_noisy[:, 1].numpy()
    las.z = x_noisy[:, 2].numpy()
    
    # Add RGB values (scale back to 16-bit)
    las.red = (rgb[:, 0].numpy() * 65535).astype(np.uint16)
    las.green = (rgb[:, 1].numpy() * 65535).astype(np.uint16)
    las.blue = (rgb[:, 2].numpy() * 65535).astype(np.uint16)
    
    # Add semantic labels
    las.classification = y.numpy()
    
    # Add SNR level and noise level as extra dimensions
    las.add_extra_dim(laspy.ExtraBytesParams(name="snr_db", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="noise_level", type=np.float32))
    las.snr_db = np.full(len(x_noisy), snr_db)
    las.noise_level = np.full(len(x_noisy), noise_level if noise_level is not None else 0.0)
    
    las.write(output_file)

    # Write to file
    if snr_db == float('inf') and noise_level is None:
        print(f"Saved clean scene to {output_file}")
    elif noise_level is not None:
        print(f"Saved noisy scene (variance {noise_level}) to {output_file}")
    else:
        print(f"Saved noisy scene (SNR {snr_db}dB) to {output_file}")


def argument_parser() -> argparse.ArgumentParser:
    """Create and return an argument parser for the SNR testing script.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    import argparse

    parser = argparse.ArgumentParser(description="Test GMM variants' robustness to noise")

    # Input/output arguments
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input directory or path to LAS folder/file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for results CSV file",
    )

    # Algorithm variant arguments
    parser.add_argument(
        "--variants",
        default="CEM",
        type=str,
        help="Comma-separated list of GMM variants to test",
    )
    parser.add_argument(
        "--grid_sizes",
        default=None,
        type=str,
        help="Comma-separated list of grid sizes for baseline comparison",
    )
    parser.add_argument(
        "--downsample_grid_size",
        default=None,
        type=float,
        help="Grid size for downsampling input points",
    )

    # Hierarchical GMM parameters
    parser.add_argument(
        "--approx_cluster_size",
        default="8192",
        type=str,
        help="Comma-separated list of approximate cluster sizes",
    )
    parser.add_argument(
        "--override_hk",
        default=None,
        type=str,
        help="Comma-separated list of hierarchy sizes to override",
    )
    parser.add_argument("--depth", default=4, type=int, help="Depth of the hierarchy")
    parser.add_argument(
        "--alpha", default=1e0, type=float, help="Regularization parameter alpha"
    )
    parser.add_argument("--tol", default=1e-2, type=float, help="Convergence tolerance")

    # Iteration control
    parser.add_argument(
        "--max_iter", default=10, type=int, help="Maximum number of EM iterations"
    )
    parser.add_argument(
        "--over_iter",
        default=0,
        type=int,
        help="Number of additional iterations after convergence",
    )

    # Feature computation parameters
    parser.add_argument(
        "--features",
        default=None,
        type=str,
        help="Comma-separated list of features to compute",
    )
    parser.add_argument(
        "--k_neighbors",
        default=50,
        type=int,
        help="Number of neighbors for kNN graph construction",
    )

    # SNR testing parameters
    parser.add_argument(
        "--snr_levels",
        default="30,20,10,0",
        type=str,
        help="Comma-separated list of SNR levels in dB to test",
    )
    parser.add_argument(
        "--noise_levels",
        default=None,
        type=str,
        help="Comma-separated list of direct noise levels (variances) to test. If provided, overrides SNR levels.",
    )
    parser.add_argument(
        "--num_noise_trials",
        default=5,
        type=int,
        help="Number of trials for each noise level",
    )

    # Debug options
    parser.add_argument(
        "--verbose", default=True, type=bool, help="Print debug information"
    )

    # Add argument for saving noisy scenes
    parser.add_argument(
        "--save_noisy_scenes",
        action="store_true",
        help="Save example scenes for each noise level",
    )

    return parser


if __name__ == "__main__":
    # Parse command line arguments
    parser = argument_parser()
    args = parser.parse_args()

    # Handle input path
    if Path(args.input).is_file():
        source_path = [args.input]
    elif Path(args.input).is_dir():
        source_path = [
            file
            for file in Path(args.input).iterdir()
            if file.is_file() and file.suffix == ".las"
        ]
    else:
        raise ValueError(f"Invalid input path: {args.input}")

    # Handle output path
    if args.output is None:
        output_path = Path(source_path[0]).parent / "snr_results.csv"
    elif Path(args.output).exists():
        raise ValueError(f"Output path already exists: {args.output}")
    else:
        output_path = args.output

    # Parse noise levels
    if args.noise_levels is not None:
        # Use direct noise levels if provided
        noise_levels = [float(level) for level in args.noise_levels.split(",")]
        # Calculate SNR levels from noise levels and signal power
        snr_levels = []
        for noise_level in noise_levels:
            if noise_level == 0:
                snr_levels.append(float('inf'))
            else:
                # We'll calculate actual SNR after loading the data
                snr_levels.append(0.0)  # Placeholder
    else:
        # Parse SNR levels
        snr_levels = []
        for snr in args.snr_levels.split(","):
            snr = snr.strip().lower()
            if snr in ['inf', 'infinity']:
                snr_levels.append(float('inf'))
            else:
                snr_levels.append(float(snr))
        noise_levels = [None] * len(snr_levels)  # Set noise_levels to None when using SNR

    # Parse downsampling grid size
    if args.downsample_grid_size is not None:
        downsample_grid_size = float(args.downsample_grid_size)

    # Parse GMM variants
    if args.variants is not None:
        variants = [
            getattr(GMMVariant, variant) for variant in args.variants.split(",")
        ]
    else:
        variants = []

    # Parse additional features
    if args.features is not None:
        features_names = [feature.strip() for feature in args.features.split(",")]
    else:
        features_names = None

    # Create hierarchies for each cluster size
    if args.override_hk is None:
        hierarchy_ks = [
            make_hierarchy(int(size), args.depth)
            for size in args.approx_cluster_size.split(",")
        ]
    else:
        hierarchy_ks = [
            [int(size) for size in args.override_hk.split(",")]
        ]

    # Parse grid sizes for baseline comparison
    if args.grid_sizes is not None:
        grid_sizes = [float(s) for s in args.grid_sizes.split(",")]
    else:
        grid_sizes = []

    # Initialize results list
    results_dfs = []

    if args.verbose:
        print("\nTest Configuration:")
        print(f"\tInput path: {source_path}")
        print(f"\tOutput path: {output_path}")
        print(f"\tDownsample grid size: {args.downsample_grid_size}")
        print(f"\tApproximate cluster sizes: {args.approx_cluster_size}")
        print(f"\tHierarchy depth: {args.depth}")
        print(f"\tNumber of neighbors: {args.k_neighbors}")
        print(f"\tGMM variants: {variants}")
        print(f"\tAdditional features: {features_names}")
        print(f"\tGrid sizes for baseline: {args.grid_sizes}")
        print(f"\tMax iterations: {args.max_iter}")
        print(f"\tTolerance: {args.tol}")
        print(f"\tAlpha: {args.alpha}")
        print(f"\tOver iterations: {args.over_iter}")
        print(f"\tSNR levels: {snr_levels}")
        print(f"\tNumber of noise trials: {args.num_noise_trials}\n")

    for file in source_path:
        """Process a LAS file with different noise levels"""
        print(f"Loading data from {file}")
        lasdata = laspy.read(file)
        print(list(lasdata.point_format.dimension_names))
        
        # Load clean data
        if args.downsample_grid_size is not None:
            start_time = time.time()
            x_clean, y, rgb = downsample_point_cloud(
                lasdata, args.downsample_grid_size
            )
            duration = time.time() - start_time
            print(f"Downsampled number of points: {x_clean.shape[0]} in {duration:.3f} seconds")
        else:
            x_clean = torch.tensor(lasdata.xyz, dtype=torch.float32, device="cpu")
            y = torch.tensor(
                lasdata.semantic_label.copy(), dtype=torch.int32, device="cpu"
            )
            rgb = (
                torch.tensor(
                    (
                        np.concatenate(
                            [lasdata.red, lasdata.green, lasdata.blue], axis=0
                        )
                        / 65535
                    ).reshape(3, -1).T,
                    dtype=torch.float32,
                    device="cpu",
                )
                / 255.0
            )

        # Compute features for clean data
        if features_names == ['pos']:
            data = Data(pos=x_clean)
            features_clean = x_clean
        elif features_names == ['pos', 'rgb']:
            data = Data(pos=x_clean, rgb=rgb)
            features_clean = torch.cat([x_clean, rgb], dim=1)
        else:
            print("Computing k-nearest neighbors...")
            start_time = time.time()
            neighbors, distances = knn_1_graph(
                x_clean,
                k=args.k_neighbors,
                r_max=float("inf"),
                batch=None,
                oversample=True,
                self_is_neighbor=False,
                verbose=args.verbose,
                trim=False,
            )
            duration = time.time() - start_time
            print(f"KNN completed in {duration:.3f} seconds")

            data = Data(
                pos=x_clean,
                rgb=rgb,
                neighbor_index=neighbors[1].view(-1, args.k_neighbors),
                neighbor_distance=distances.view(-1, args.k_neighbors),
            )
            
            if 'direction' in features_names:
                print("Computing direction features...")
                start_time = time.time()
                directions = direction_features(
                    x_clean, 
                    data.neighbor_index, 
                )
                data.direction = directions
                duration = time.time() - start_time
                print(f"Direction features computed in {duration:.3f} seconds")
            
            print("Computing features...")
            start_time = time.time()
            to_compute = [feature for feature in features_names if feature not in ['pos', 'rgb', 'direction']]
            if to_compute:
                point_features = PointFeatures(
                    keys=to_compute, k_min=5, k_step=-1, overwrite=True
                )
                data = point_features(data)
            duration = time.time() - start_time
            print(f"Geometric features computed in {duration:.3f} seconds")

            if "elevation" in features_names:
                print("Computing elevation...")
                start_time = time.time()
                ground_elevation = GroundElevation(
                    z_threshold=1.5, xy_grid=None, model="ransac", scale=4.0
                )
                data = ground_elevation(data)
                duration = time.time() - start_time
                print(f"Elevation computed in {duration:.3f} seconds")

            features_clean = torch.cat(
                [getattr(data, feature_name) for feature_name in features_names],
                dim=1,
            )

        print("\nData is ready for testing:")
        print(f" - Number of points: {x_clean.shape[0]}" + (" (downsampled)" if args.downsample_grid_size is not None else ""))
        print(f" - Dimensions: {features_clean.shape}")
        print(f" - Device: {features_clean.device}")

        # Test each noise level
        for snr_db, noise_level in zip(snr_levels, noise_levels):
            print(f"\nTesting {'noise level' if noise_level is not None else 'SNR level'}: {noise_level if noise_level is not None else f'{snr_db} dB'}")
            
            # Run multiple trials for each noise level
            for trial in range(args.num_noise_trials):
                print(f"\nTrial {trial + 1}/{args.num_noise_trials}")
                
                # Add noise to point cloud
                x_noisy, noise_variance = add_gaussian_noise(x_clean, snr_db, noise_level)
                
                # If using noise levels, calculate actual SNR
                if noise_level is not None:
                    snr_db = compute_snr(x_clean, x_noisy)
                    print(f"Computed SNR: {snr_db:.2f} dB")
                
                # Save noisy scene if requested
                if args.save_noisy_scenes and trial == 0:  # Save only first trial for each noise level
                    save_noisy_scene(
                        x_noisy=x_noisy,
                        y=y,
                        rgb=rgb,
                        snr_db=snr_db,
                        noise_level=noise_level,
                        original_file=Path(file),
                        trial=trial
                    )
                
                # Compute features for noisy data
                if features_names == ['pos']:
                    features_noisy = x_noisy
                elif features_names == ['pos', 'rgb']:
                    features_noisy = torch.cat([x_noisy, rgb], dim=1)
                else:
                    # Recompute features with noisy coordinates
                    data_noisy = Data(pos=x_noisy)
                    if 'rgb' in features_names:
                        data_noisy.rgb = rgb
                    
                    if 'direction' in features_names:
                        directions = direction_features(
                            x_noisy, 
                            data.neighbor_index, 
                        )
                        data_noisy.direction = directions
                    
                    to_compute = [feature for feature in features_names if feature not in ['pos', 'rgb', 'direction']]
                    if to_compute:
                        point_features = PointFeatures(
                            keys=to_compute, k_min=5, k_step=-1, overwrite=True
                        )
                        data_noisy = point_features(data_noisy)
                    
                    if "elevation" in features_names:
                        ground_elevation = GroundElevation(
                            z_threshold=1.5, xy_grid=None, model="ransac", scale=4.0
                        )
                        data_noisy = ground_elevation(data_noisy)
                    
                    features_noisy = torch.cat(
                        [getattr(data_noisy, feature_name) for feature_name in features_names],
                        dim=1,
                    )

                # Test GMM variants
                for variant in variants:
                    for hierarchy_k in hierarchy_ks:
                        actual_size = math.prod(hierarchy_k)
                        print(f"\nTesting {variant} with hierarchy_k: {hierarchy_k} ({actual_size}):")
                        start_time = time.time()
                        result_dict, predictions, error, cluster, mu, sigma = test_em(
                            x=features_noisy,
                            y=y,
                            hierarchy_k=hierarchy_k,
                            alpha=args.alpha,
                            tol=args.tol,
                            max_iter=args.max_iter,
                            variant=variant,
                            over_iter=args.over_iter,
                            noise_variance=noise_level if noise_level is not None else noise_variance,
                        )
                        duration = time.time() - start_time
                        result_dict["duration"] = duration
                        result_dict["filename"] = file
                        result_dict["point_count"] = features_noisy.shape[0]
                        result_dict["variant"] = variant.name
                        result_dict["actual_size"] = actual_size
                        result_dict["depth"] = args.depth
                        result_dict["snr_db"] = snr_db
                        result_dict["noise_variance"] = noise_level if noise_level is not None else noise_variance
                        result_dict["trial"] = trial
                        results_dfs.append(pd.DataFrame(result_dict))

                # Test baseline grid partitioning
                if grid_sizes is not None:
                    for grid_size in grid_sizes:
                        print(f"\nTesting baseline grid partitioning with size {grid_size}:")
                        result_dict = baseline(x_noisy, y, grid_size, file, noise_variance)
                        result_dict["snr_db"] = snr_db
                        result_dict["trial"] = trial
                        results_dfs.append(pd.DataFrame(result_dict))

    # Create DataFrame and save to CSV
    results_df = pd.concat(results_dfs)

    # Reorder columns for better readability
    column_order = [
        "filename",
        "variant",
        "point_count",
        "accuracy",
        "miou",
        "duration",
        "iterations",
        "actual_size",
        "depth",
        "snr_db",
        "noise_variance",
        "trial",
    ]
    results_df = results_df[column_order]
    results_df.to_csv(output_path, index=False)

    # Print summary
    print("\nResults Summary:")
    print(results_df.to_string())
    print(f"\nResults saved to: {output_path}") 