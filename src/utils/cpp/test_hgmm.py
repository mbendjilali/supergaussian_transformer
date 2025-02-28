import torch
import time
import os
import laspy
import pandas as pd
from gaussian_mixture_cpp import hierarchical_gmm, GMMVariant
from pathlib import Path
import math
from pandarallel import pandarallel
pandarallel.initialize()      

def batch_sqrt_lower_triangular(L):
    S = torch.zeros_like(L)
    S[:, 0, 0] = torch.sqrt(L[:, 0, 0])
    S[:, 1, 0] = L[:, 1, 0] / S[:, 0, 0]
    S[:, 1, 1] = torch.sqrt(L[:, 1, 1] - S[:, 1, 0]**2)
    S[:, 2, 0] = L[:, 2, 0] / S[:, 0, 0]
    S[:, 2, 1] = (L[:, 2, 1] - S[:, 1, 0] * S[:, 2, 0]) / S[:, 1, 1]
    S[:, 2, 2] = torch.sqrt(L[:, 2, 2] - S[:, 2, 0]**2 - S[:, 2, 1]**2)
    return S

def hard_assign(labels, y):
    # Convert to pandas series
    labels_series = pd.Series(labels.cpu().numpy())
    y_series = pd.Series(y.cpu().numpy())
    predictions = labels_series.groupby(labels_series).transform(lambda x: y_series[x.index].mode()[0])  
    # Convert back to torch tensor on same device as input
    return torch.tensor(predictions.values, device=labels.device)

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
        # Get mask for current class
        mask = (y == cls)
        if mask.sum() == 0:
            continue
            
        # Compute accuracy for this class
        class_acc = (predictions[mask] == y[mask]).float().mean()
        accuracies.append(class_acc)
    
    # Return mean over all classes
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
    # Initialize intersection and union arrays
    intersections = torch.zeros(num_classes, device=y.device)
    unions = torch.zeros(num_classes, device=y.device)
    
    # Compute IoU for each class
    for cls in range(num_classes):
        # Get binary masks for current class
        pred_mask = (predictions == cls)
        true_mask = (y == cls)
        
        # Compute intersection and union
        intersection = (pred_mask & true_mask).sum().float()
        union = (pred_mask | true_mask).sum().float()
        
        # Update arrays
        intersections[cls] = intersection
        unions[cls] = union
    
    # Compute IoU for classes that appear in the ground truth
    valid_classes = unions > 0
    if valid_classes.sum() == 0:
        return 0.0
        
    ious = intersections[valid_classes] / unions[valid_classes]
    return ious.mean().item()

def downsample_point_cloud(x, y, grid_size):
    """Downsample point cloud using voxel grid."""
    min_coords = x.min(dim=0)[0]
    v = ((x - min_coords) / grid_size).long()
    
    # Create unique voxel IDs
    voxel_ids = (
        (v[:, 0] + v[:, 1] + v[:, 2]) * 
        (v[:, 0] + v[:, 1] + v[:, 2] + 1) * 
        (v[:, 0] + v[:, 1] + v[:, 2] + 2) // 6 + 
        (v[:, 1] + v[:, 2]) * (v[:, 1] + v[:, 2] + 1) // 2 + 
        v[:, 2]
    )
    
    # Get unique voxels and mapping
    unique_ids, inverse_indices = voxel_ids.unique(return_inverse=True)
    
    # Use scatter_mean for efficient centroid computation
    ones = torch.ones_like(voxel_ids, dtype=torch.float)
    counts = torch.zeros(len(unique_ids), device=x.device)
    counts.scatter_add_(0, inverse_indices, ones)
    
    x_downsampled = torch.zeros((len(unique_ids), 3), device=x.device)
    for dim in range(3):
        x_downsampled[:, dim].scatter_add_(0, inverse_indices, x[:, dim])
    x_downsampled /= counts.unsqueeze(1)
    
    # Get most common label per voxel using the existing hard_assign function
    y_downsampled = hard_assign(inverse_indices, y)
    
    return x_downsampled, y_downsampled, inverse_indices

def test_em(
    filename,
    x, 
    y,
    hierarchy_k, 
    alpha, 
    tol, 
    max_iter, 
    variant,
    over_iter,
    downsample_grid_size=None,
):
    # Convert hierarchy_k to tensor
    hierarchy_k_tensor = torch.tensor(hierarchy_k, dtype=torch.long, device=x.device)
    print(f"\nRunning Hierarchical GMM with variant: {variant}")
    
    results_dict = {
        "duration": [], 
        "accuracy": [],
        "miou": [],
        "iterations": [],
        "point_count": x.shape[0],
        "filename": filename
    }
    
    for i in range(over_iter + 1):
        start_time = time.time()
        
        # If downsampling is enabled, run EM on downsampled points
        if downsample_grid_size is not None:
            x_down, y_down, inverse_indices = downsample_point_cloud(x, y, downsample_grid_size)
            all_level_labels, _, _, _ = hierarchical_gmm(
                x_down,
                hierarchy_k_tensor,
                alpha,
                tol,
                max_iter + i,
                variant
            )
            # Map labels back to original points
            predictions = hard_assign(all_level_labels[-1][inverse_indices], y)
        else:
            all_level_labels, _, _, _ = hierarchical_gmm(
                x,
                hierarchy_k_tensor,
                alpha,
                tol,
                max_iter + i,
                variant
            )
            predictions = hard_assign(all_level_labels[-1], y)
            
        duration = time.time() - start_time
        print(f"{max_iter + i} iterations completed in {duration:.3f} seconds")
        
        start_time = time.time()
        accuracy = compute_mean_accuracy(predictions, y)
        duration = time.time() - start_time
        print(f"Accuracy: {accuracy:.3f} in {duration:.3f} seconds")

        start_time = time.time()    
        miou = compute_miou(predictions, y, num_classes=13)  # Assuming 13 classes for example
        duration = time.time() - start_time
        print(f"mIoU: {miou:.3f} in {duration:.3f} seconds")
        
        results_dict["duration"].append(duration)
        results_dict["iterations"].append(max_iter + i)
        results_dict["accuracy"].append(accuracy)
        results_dict["miou"].append(miou)
        
    return results_dict

def make_hierarchy(approx_cluster_size, depth):
    """
    Generate a hierarchy of integers whose product approximates the target cluster size.
    
    Args:
        approx_cluster_size (int): Target cluster size to approximate
        depth (int): Number of levels in the hierarchy
    
    Returns:
        list[int]: List of integers in descending order whose product is the nearest
                   power of 2 less than or equal to approx_cluster_size
    """
    # Find the nearest power of 2 less than or equal to approx_cluster_size
    target = 2 ** (approx_cluster_size.bit_length() - 1)
    if target > approx_cluster_size:
        target >>= 1
    
    # Find the nth root rounded down to nearest power of 2
    base = 2 ** ((target.bit_length() - 1) // depth)
    
    # Initialize result with base values
    result = [base] * depth
    
    # Distribute remaining factors (always powers of 2) to maximize first elements
    remaining_power = (target // (base ** depth)).bit_length() - 1
    for i in range(remaining_power):
        idx = i % depth
        result[idx] *= 2
    
    # Sort in descending order
    result.sort(reverse=True)
    return result

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
        "duration": [],
        "accuracy": [],
        "miou": [],
        "iterations": [],
        "point_count": x.shape[0],
        "filename": filename,
        "variant": "GRID",
        "actual_size": 0,  # Will be updated with actual number of non-empty voxels
        "depth": 1
    }
    start_time = time.time()
    # Voxelize points
    min_coords = x.min(dim=0)[0]
    v = ((x - min_coords) / grid_size).long()

    # Create unique voxel IDs
    labels = (
        (v[:, 0] + v[:, 1] + v[:, 2]) * 
        (v[:, 0] + v[:, 1] + v[:, 2] + 1) * 
        (v[:, 0] + v[:, 1] + v[:, 2] + 2) // 6 + 
        (v[:, 1] + v[:, 2]) * (v[:, 1] + v[:, 2] + 1) // 2 + 
        v[:, 2]
    )
    predictions = hard_assign(labels, y)
    duration = time.time() - start_time
    # Get number of non-empty voxels 
    num_voxels = labels.unique().numel()
    accuracy = compute_mean_accuracy(predictions, y)
    miou = compute_miou(predictions, y, num_classes=13)  # Assuming 13 classes for example
    print(f"Number of non-empty voxels: {num_voxels}, Accuracy: {accuracy:.3f}, mIoU: {miou:.3f} in {duration:.3f} seconds")
    
    # Fill results dictionary
    results_dict["duration"].append(duration)
    results_dict["accuracy"].append(accuracy)
    results_dict["miou"].append(miou)
    results_dict["iterations"].append(1)  # Grid partitioning is non-iterative
    results_dict["actual_size"] = num_voxels
    
    return results_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test Hierarchical GMM variants')
    parser.add_argument('--input', type=str, required=True, help='Input directory or path to LAS folder/file')
    parser.add_argument('--downsample_grid_size', type=float, help='Downsample grid size')
    parser.add_argument('--output', type=str, help='Output path for results CSV file')
    parser.add_argument('--variants', type=str, help='Variants to test')
    parser.add_argument('--approx_cluster_size', type=str, default="8192", help='Approximate cluster size')
    parser.add_argument('--depth', type=int, default=3, help='Depth of the hierarchy')
    parser.add_argument('--alpha', type=float, default=1e1, help='Alpha')
    parser.add_argument('--tol', type=float, default=1.0, help='Tolerance')
    parser.add_argument('--max_iter', type=int, default=1, help='Maximum iterations')
    parser.add_argument('--over_iter', type=int, default=0, help='Over iterations')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    parser.add_argument('--grid_sizes', type=str, 
                       help='Comma-separated list of grid sizes for baseline comparison')
    args = parser.parse_args()

    input = args.input
    output = args.output
    variants = args.variants
    approx_cluster_size = args.approx_cluster_size
    depth = args.depth
    alpha = args.alpha
    tol = args.tol
    max_iter = args.max_iter
    over_iter = args.over_iter
    verbose = args.verbose
    grid_sizes = args.grid_sizes
    downsample_grid_size = args.downsample_grid_size

    if Path(input).is_file():
        source_path = [input]
    elif Path(input).is_dir():
        source_path = os.listdir(input)
    else:
        raise ValueError(f"Invalid input path: {input}")
    
    if downsample_grid_size is not None:
        downsample_grid_size = float(downsample_grid_size)
    else:
        downsample_grid_size = None
    
    if output is None:
        output_path = Path(source_path[0]).parent / "results.csv"
    elif Path(output).exists():
        raise ValueError(f"Output path already exists: {output}")
    else:
        output_path = output
    
    if variants is not None:
        variants = variants.split(",")
        variants = [getattr(GMMVariant, variant) for variant in variants]
    else:
        variants = []
    
    # Generate hierarchy_ks using the new function
    hierarchy_ks = [make_hierarchy(int(size), depth) for size in approx_cluster_size.split(",")]
    
    if grid_sizes is not None:
        grid_sizes = [float(s) for s in grid_sizes.split(",")]
    else:
        grid_sizes = []
    
    results_dfs = []
    
    for file in source_path:
        """Process a LAS file with Hierarchical GMM and save results"""
        print(f"Loading data from {os.path.join(input, file)}")
        
        # Read LAS file
        lasdata = laspy.read(os.path.join(input, file))
        # Load points and scale them
        x = torch.tensor(lasdata.xyz, dtype=torch.float32, device="cpu")
        y = torch.tensor(lasdata.semantic_label.copy(), dtype=torch.int32, device="cpu")
        print("\nInput data info:")
        print(f" - Number of points: {x.shape[0]}")
        print(f" - Dimensions: {x.shape[1]}")
        print(f" - Device: {x.device}")
        if downsample_grid_size is not None:
            # TODO: Downsample the point cloud
            print(f" - Accuracy metrics are reported for the full, unsampled point cloud.")
        else:
            # TODO ?
            None
        for variant in variants:
            for hierarchy_k in hierarchy_ks:
                actual_size = math.prod(hierarchy_k)
                print(f"\nTesting {variant} on {file} with hierarchy_k: {hierarchy_k} ({actual_size}):")
                result_dict = test_em(
                    x=x,
                    y=y,
                    hierarchy_k=hierarchy_k,
                    alpha=alpha,
                    tol=tol,
                    max_iter=max_iter,
                    variant=variant,
                    over_iter=over_iter,
                    downsample_grid_size=downsample_grid_size,
                    filename=file,
                )
                # Add variant information to result dictionary
                result_dict["variant"] = variant.name
                result_dict["actual_size"] = actual_size
                result_dict["depth"] = depth
                results_dfs.append(pd.DataFrame(result_dict))

        # Add baseline results if grid sizes are provided
        if grid_sizes is not None:
            for grid_size in grid_sizes:
                print(f"\nTesting baseline grid partitioning with size {grid_size}:")
                result_dict = baseline(x, y, grid_size, file)
                results_dfs.append(pd.DataFrame(result_dict))

    # Create DataFrame and save to CSV
    results_df = pd.concat(results_dfs)
    
    # Reorder columns for better readability
    column_order = ['filename', 'variant', 'point_count', 'accuracy', 'miou', 'duration', 'iterations', 'actual_size', 'depth']
    results_df = results_df[column_order]
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\nResults Summary:")
    print(results_df.to_string())
    print(f"\nResults saved to: {output_path}")