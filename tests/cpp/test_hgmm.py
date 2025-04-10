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


def batch_sqrt_lower_triangular(L):
    S = torch.zeros_like(L)
    S[:, 0, 0] = torch.sqrt(L[:, 0, 0])
    S[:, 1, 0] = L[:, 1, 0] / S[:, 0, 0]
    S[:, 1, 1] = torch.sqrt(L[:, 1, 1] - S[:, 1, 0] ** 2)
    S[:, 2, 0] = L[:, 2, 0] / S[:, 0, 0]
    S[:, 2, 1] = (L[:, 2, 1] - S[:, 1, 0] * S[:, 2, 0]) / S[:, 1, 1]
    S[:, 2, 2] = torch.sqrt(L[:, 2, 2] - S[:, 2, 0] ** 2 - S[:, 2, 1] ** 2)
    return S


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


def hard_assign(labels, y):
    """
    Vectorized implementation to assign most common y value to each label using one-hot encoding.
    
    Args:
        labels: Cluster assignment for each point (N,)
        y: Ground truth labels (N,)
        
    Returns:
        Tensor: Most common y value for each cluster
    """
    # Ensure inputs are properly typed
    labels = labels.long()
    y = y.long()
    
    # Get unique labels and number of classes
    unique_labels = labels.unique()
    num_classes = y.max().item() + 1
    
    # Create mapping from original labels to contiguous indices
    label_to_idx = torch.zeros(unique_labels.max().item() + 1, dtype=torch.long, device=labels.device)
    label_to_idx[unique_labels] = torch.arange(len(unique_labels), device=labels.device)
    
    # Map input labels to contiguous indices
    label_idx = label_to_idx[labels]
    
    # Create vote matrix: rows are labels, columns are classes
    votes = torch.zeros((len(unique_labels), num_classes), device=labels.device)
    
    # This is the key vectorized operation - add 1 vote for each (label,class) pair
    votes.index_put_((label_idx, y), torch.ones(len(y), device=labels.device), accumulate=True)
    
    # Find most common class for each label
    most_common_classes = votes.argmax(dim=1)
    
    # Map back to original points using efficient indexing
    result = most_common_classes[label_idx]
    
    return result


def compute_mean_accuracy(predictions, y):
    """Compute mean class accuracy for semantic segmentation.

    Args:
        predictions: Predicted labels (N,)
        y: Ground truth labels (N,)

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


def compute_miou(predictions, y, num_classes):
    """Compute mean IoU for semantic segmentation.

    Args:
        predictions: Predicted labels (N,)
        y: Ground truth labels (N,)
        num_classes: Number of semantic classes

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


def test_em(
    x,
    y,
    hierarchy_k,
    alpha,
    tol,
    max_iter,
    variant,
):
    # Convert hierarchy_k to tensor
    hierarchy_k = torch.tensor(hierarchy_k, dtype=torch.long, device=x.device)
    results_dict = {
        "duration": 0.0,
        "accuracy": 0.0,
        "miou": 0.0,
        "iterations": 0,
    }

    start_time = time.time()
    cluster, _, mu, sigma = hierarchical_gmm(
        x, hierarchy_k, alpha, tol, max_iter, variant
    )
    duration = time.time() - start_time
    print(f"Duration: {duration:.3f} seconds")

    predictions = hard_assign(cluster[-1], y)
    error = (predictions != y).int()
    accuracy = compute_mean_accuracy(predictions, y)

    print(f"Mean class accuracy: {accuracy:.3f}")
    miou = compute_miou(
        predictions, y, num_classes=13
    )  # Assuming cluster_size classes for example
    print(f"Mean IoU: {miou:.3f}")

    results_dict["duration"] = duration
    results_dict["iterations"] = max_iter
    results_dict["accuracy"] = accuracy
    results_dict["miou"] = miou

    return results_dict, predictions, error, cluster, mu, sigma


def baseline(x, y, grid_size, filename):
    """Compute baseline metrics using voxel grid partitioning.

    Args:
        x: Input points (N, 3)
        y: Ground truth labels (N,)
        grid_size: Size of voxel grid
        filename: Name of input file for logging

    Returns:
        dict: Results dictionary with same format as test_em
    """
    # Initialize results dictionary
    results_dict = {
        "duration": 0,
        "accuracy": 0,
        "miou": 0,
        "iterations": 1,
        "point_count": x.shape[0],
        "filename": filename,
        "variant": "GRID",
        "actual_size": 0,  # Will be updated with actual number of non-empty voxels
        "depth": 1,
    }
    start_time = time.time()
    # Voxelize points
    min_coords = x.min(dim=0)[0]
    v = ((x - min_coords) / grid_size).long()

    # Create unique voxel IDs
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
    # Get number of non-empty voxels
    num_voxels = labels.unique().numel()
    accuracy = compute_mean_accuracy(predictions, y)
    miou = compute_miou(
        predictions, y, num_classes=13
    )  # Assuming 13 classes for example

    # Fill results dictionary
    results_dict["duration"] = duration
    results_dict["accuracy"] = accuracy
    results_dict["miou"] = miou
    results_dict["iterations"] = 1  # Grid partitioning is non-iterative
    results_dict["actual_size"] = num_voxels

    return results_dict


def direction_features(pos, neighbors):    
    # Get positions of neighbors
    n_points = pos.shape[0]
    neighbor_pos = pos[neighbors.view(-1)].view(n_points, -1, 3)
    
    # Center points
    centers = pos.unsqueeze(1)
    centered = neighbor_pos - centers
    
    # Compute covariance matrices for each point
    # (N, 3, 3) covariance matrices
    cov = torch.bmm(centered.transpose(1, 2), centered) / (neighbors.shape[1] - 1)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    max_values, max_indices = torch.max(eigenvalues, dim=1)
    directions = max_values[:, None] * eigenvectors[torch.arange(n_points), max_indices]
    
    return directions


def argument_parser():
    """Create and return an argument parser for the Hierarchical GMM testing script."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Hierarchical GMM variants")

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
        "--cluster_size",
        default="cluster_size192",
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
        "--max_iter", default=5, type=int, help="Maximum number of EM iterations"
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

    # Debug options
    parser.add_argument(
        "--verbose", default=True, type=bool, help="Print debug information"
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
        output_path = Path(source_path[0]).parent / "results.csv"
    elif Path(args.output).exists():
        raise ValueError(f"Output path already exists: {args.output}")
    else:
        output_path = args.output

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
                for size in args.cluster_size.split(",")
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
        print(f"\tApproximate cluster sizes: {args.cluster_size}")
        print(f"\tHierarchy depth: {args.depth}")
        print(f"\tNumber of neighbors: {args.k_neighbors}")
        print(f"\tGMM variants: {variants}")
        print(f"\tAdditional features: {features_names}")
        print(f"\tGrid sizes for baseline: {args.grid_sizes}")
        print(f"\tMax iterations: {args.max_iter}")
        print(f"\tTolerance: {args.tol}")
        print(f"\tAlpha: {args.alpha}")

    for file in source_path:
        """Process a LAS file with Hierarchical GMM and save results"""
        print(f"Loading data from {file}")
        lasdata = laspy.read(file)
        if args.downsample_grid_size is not None:
            start_time = time.time()
            x, y, rgb = downsample_point_cloud(
                lasdata, args.downsample_grid_size
            )
            duration = time.time() - start_time
            print(
                f"Downsampled number of points: {x.shape[0]} in {duration:.3f} seconds"
            )
            print(y.shape)
        else:
            x = torch.tensor(lasdata.xyz, dtype=torch.float32, device="cpu")

            y = torch.tensor(
                lasdata.classification.copy(), dtype=torch.int32, device="cpu"
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
        if features_names == ['pos']:
            data = Data(pos=x)
            features = x
        elif features_names == ['pos', 'rgb']:
            data = Data(pos=x, rgb=rgb)
            features = torch.cat([x, rgb], dim=1)
        else:
            print("Computing k-nearest neighbors...")
            start_time = time.time()
            # First, get the k-nearest neighbors for feature computation
            neighbors, distances = knn_1_graph(
                x,
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

            # Create initial Data object
            data = Data(
                pos=x,
                rgb=rgb,
                neighbor_index=neighbors[1].view(-1, args.k_neighbors),
                neighbor_distance=distances.view(-1, args.k_neighbors),
            )
            
            # Compute direction features if requested
            if 'direction' in features_names:
                print("Computing direction features...")
                start_time = time.time()
                directions = direction_features(
                    x, 
                    data.neighbor_index, 
                )
                data.direction = directions
                duration = time.time() - start_time
                print(f"Direction features computed in {duration:.3f} seconds")
            
            print("Computing features...")
            start_time = time.time()
            # Compute geometric features
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
                # Compute elevation
                ground_elevation = GroundElevation(
                    z_threshold=1.5, xy_grid=None, model="ransac", scale=4.0
                )
                data = ground_elevation(data)
                duration = time.time() - start_time
                print(f"Elevation computed in {duration:.3f} seconds")

            features = torch.cat(
                [getattr(data, feature_name) for feature_name in features_names],
                dim=1,
            )

        print("\nData is ready for testing:")
        print(f" - Number of points: {x.shape[0]}" + (" (downsampled)" if args.downsample_grid_size is not None else ""))
        print(f" - Dimensions: {features.shape}")
        print(f" - Device: {features.device}")

        for variant in variants:
            for hierarchy_k in hierarchy_ks:
                actual_size = math.prod(hierarchy_k)
                print(
                    f"\nTesting {variant} on {file} with hierarchy_k: {hierarchy_k} ({actual_size}):"
                )
                result_dict, predictions, error, cluster, mu, sigma = test_em(
                    x=features,
                    y=y,
                    hierarchy_k=hierarchy_k,
                    alpha=args.alpha,
                    tol=args.tol,
                    max_iter=args.max_iter,
                    variant=variant,
                )
                result_dict["filename"] = file
                result_dict["point_count"] = features.shape[0]
                result_dict["variant"] = variant.name
                result_dict["actual_size"] = cluster[-1].unique().shape[0]
                result_dict["depth"] = args.depth
                print(result_dict)
                results_dfs.append(pd.DataFrame(result_dict, index=[0]))

                # Save preprocessed data as LAS file
                preprocessed_path = Path(file).parent / f"preprocessed_{Path(file).name}"
                print(f"\nSaving preprocessed data to {preprocessed_path}")
                
                # Create new LAS file with same header as input
                preprocessed_las = laspy.LasData(laspy.LasHeader(version="1.4", point_format=7))
                
                # Add core XYZ coordinates
                preprocessed_las.x = data.pos[:, 0].numpy()
                preprocessed_las.y = data.pos[:, 1].numpy()
                preprocessed_las.z = data.pos[:, 2].numpy()
                
                # Add RGB values (scale back to 16-bit)
                if "rgb" in features_names:
                    preprocessed_las.red = data.rgb[:, 0].numpy().astype(np.uint16)
                    preprocessed_las.green = data.rgb[:, 1].numpy().astype(np.uint16)
                    preprocessed_las.blue = data.rgb[:, 2].numpy().astype(np.uint16)
                
                # Add semantic labels
                preprocessed_las.classification = y.numpy()
                preprocessed_las.add_extra_dim(laspy.ExtraBytesParams(name="predictions", type=np.int32))
                preprocessed_las.predictions = predictions.numpy()
                preprocessed_las.add_extra_dim(laspy.ExtraBytesParams(name="error", type=np.int32))
                preprocessed_las.error = error.numpy()
                preprocessed_las.add_extra_dim(laspy.ExtraBytesParams(name="cluster_0", type=np.int32))
                preprocessed_las.cluster_0 = cluster[0].numpy()
                preprocessed_las.add_extra_dim(laspy.ExtraBytesParams(name="cluster_1", type=np.int32))
                preprocessed_las.cluster_1 = cluster[1].numpy()
                preprocessed_las.add_extra_dim(laspy.ExtraBytesParams(name="cluster_2", type=np.int32))
                preprocessed_las.cluster_2 = cluster[2].numpy()
                
                # Add remaining feature dimensions as extra bytes
                for feature_name in features_names:
                    if feature_name == "rgb" or feature_name == "pos":
                        continue
                    elif feature_name == "direction":
                        preprocessed_las.add_extra_dim(laspy.ExtraBytesParams(name=feature_name + "_x", type=np.float32))
                        preprocessed_las.direction_x = data.direction[:, 0].numpy()
                        preprocessed_las.add_extra_dim(laspy.ExtraBytesParams(name=feature_name + "_y", type=np.float32))
                        preprocessed_las.direction_y = data.direction[:, 1].numpy()
                        preprocessed_las.add_extra_dim(laspy.ExtraBytesParams(name=feature_name + "_z", type=np.float32))
                        preprocessed_las.direction_z = data.direction[:, 2].numpy()
                    else:
                        preprocessed_las.add_extra_dim(laspy.ExtraBytesParams(name=feature_name, type=np.float32))
                        setattr(preprocessed_las, feature_name, getattr(data, feature_name).squeeze().numpy())
                
                # Write to file
                preprocessed_las.write(preprocessed_path)

                # Save Gaussian parameters
                gaussian_path = Path(file).parent / f"gaussians_{Path(file).name.replace('.las', '.npz')}"
                print(f"Saving Gaussian parameters to {gaussian_path}")
                
                # Create a dictionary to store all parameters
                gaussian_dict = {}

                # Store each level's parameters separately
                for level in range(len(mu)):
                    gaussian_dict[f'mu_level_{level}'] = mu[level].cpu().numpy()
                    gaussian_dict[f'sigma_level_{level}'] = sigma[level].cpu().numpy()

                # Save as npz file (compressed numpy format)
                np.savez_compressed(gaussian_path, **gaussian_dict)
                print(f"Gaussian parameters saved with {len(mu)} levels")

                


        # Add baseline results if grid sizes are provided
        if grid_sizes is not None:
            for grid_size in grid_sizes:
                result_dict = baseline(x, y, grid_size, file)
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
    ]
    results_df = results_df[column_order]
    results_df.to_csv(output_path, index=False)

    # Print summary
    print("\nResults Summary:")
    print(results_df.to_string())
    print(f"\nResults saved to: {output_path}")
