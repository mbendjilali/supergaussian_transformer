# Standard library imports
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
from voxel_partition_cpp import voxelize_points_with_size

# Add necessary paths for imports
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(
    osp.dirname(current_dir)
)  # Go up 2 levels to reach project root
sys.path.append(project_root)  # Add project root to Python path


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
    
    # Create mapping from original labels to contiguous indices using a dictionary
    label_to_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}
    
    # Map input labels to contiguous indices using list comprehension
    label_idx = torch.tensor([label_to_idx[label.item()] for label in labels], 
                           dtype=torch.long, device=labels.device)
    
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


def test_voxel_partition(x, y, voxel_size):
    """Test voxel partitioning functionality.
    
    Args:
        x: Input points (N, 3)
        y: Ground truth labels (N,)
        voxel_size: List of voxel sizes
        
    Returns:
        dict: Results dictionary with metrics
    """
    
    # Initialize results dictionary
    results_dict = {
        "duration": 0.0,
        "accuracy": 0.0,
        "miou": 0.0,
        "num_voxels": 0,
    }
    
    # Run voxel partitioning - ensure voxel_size is a float/double
    start_time = time.time()

    
    # Process each voxel size individually
    num_points = x.shape[0]
    num_levels = len(voxel_size)
    labels = torch.zeros((num_points, num_levels), dtype=torch.int64)
    
    for i, vs in enumerate(voxel_size):
        # Process this level
        level_labels = voxelize_points_with_size(x, float(vs))
        labels[:, i] = level_labels
        
        # For levels > 0, assign parents (in Python)
        if i > 0:
            # For each unique label in current level
            for label in torch.unique(level_labels):
                # Find points with this label
                mask = level_labels == label
                # Get parents for these points
                parents = labels[mask, i-1]
                # If multiple parents, pick the most frequent one
                if len(torch.unique(parents)) > 1:
                    unique_parents, counts = torch.unique(parents, return_counts=True)
                    most_common_parent = unique_parents[torch.argmax(counts)]
                    # Assign this parent to all points with this label
                    labels[mask, i] = most_common_parent
    
    duration = time.time() - start_time
    print(f"Voxel partitioning completed in {duration:.3f} seconds")
    
    # Get number of unique clusters
    num_voxels = labels[:, -1].unique().numel()
    print(f"Number of voxels at last level: {num_voxels}")
    
    # Compute prediction by majority vote
    predictions = hard_assign(labels[:, -1], y)
    error = (predictions != y).int()
    
    # Compute metrics
    accuracy = compute_mean_accuracy(predictions, y)
    print(f"Mean class accuracy: {accuracy:.3f}")
    
    miou = compute_miou(predictions, y, num_classes=13)  # Assuming 13 classes
    print(f"Mean IoU: {miou:.3f}")
    
    # Update results
    results_dict["duration"] = duration
    results_dict["accuracy"] = accuracy
    results_dict["miou"] = miou
    results_dict["num_voxels"] = num_voxels
    
    return results_dict, predictions, error, labels

def argument_parser():
    """Create and return an argument parser for the Voxel Partitioning testing script."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Voxel Partitioning")

    # Input/output arguments
    parser.add_argument(
        "--input",
        # required=True,
        type=str,
        default="/home/moussabendjilali/toy_scene.las",
        help="Input directory or path to LAS folder/file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for results CSV file",
    )
    
    # Voxel partitioning parameters
    parser.add_argument(
        "--voxel_size",
        default=[0.5, 0.1, 0.05],
        type=list,
        help="Voxel size",
    )
    
    # Verbose option
    parser.add_argument(
        "--verbose", 
        default=True, 
        type=bool, 
        help="Print debug information"
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
        output_path = Path(source_path[0]).parent / "voxel_results.csv"
    else:
        output_path = args.output

    # Initialize results list
    results_dfs = []

    if args.verbose:
        print("\nTest Configuration:")
        print(f"\tInput path: {source_path}")
        print(f"\tOutput path: {output_path}")
        print(f"\tVoxel size: {args.voxel_size}")

    for file in source_path:
        print(f"Loading data from {file}")
        lasdata = laspy.read(file)
        
        # Preprocess data
        x = torch.tensor(lasdata.xyz, dtype=torch.float32, device="cpu")
        y = torch.tensor(lasdata.classification.copy(), dtype=torch.int32, device="cpu")
        
        print("\nData ready for testing:")
        print(f" - Number of points: {x.shape[0]}")
        print(f" - Device: {x.device}")
        
        # Test each voxel configuration
        print(f"\nTesting voxel partitioning with configuration: {args.voxel_size}")
            
        # Run test
        result_dict, predictions, error, labels = test_voxel_partition(
            x=x,
            y=y,
            voxel_size=args.voxel_size
        )

        # Add metadata
        result_dict["filename"] = file
        result_dict["point_count"] = x.shape[0]
            
        # Add to results
        results_dfs.append(pd.DataFrame(result_dict, index=[0]))
        
        # Save preprocessed data as LAS file
        preprocessed_path = Path(file).parent / f"voxel_preprocessed_{Path(file).name}"
        print(f"\nSaving preprocessed data to {preprocessed_path}")
        
        # Create new LAS file with same header as input
        preprocessed_las = laspy.LasData(laspy.LasHeader(version="1.4", point_format=7))
        
        # Add core XYZ coordinates
        preprocessed_las.x = x[:, 0].numpy()
        preprocessed_las.y = x[:, 1].numpy()
        preprocessed_las.z = x[:, 2].numpy()

        # Add semantic labels and predictions
        preprocessed_las.classification = y.numpy()
        preprocessed_las.add_extra_dim(laspy.ExtraBytesParams(name="predictions", type=np.int32))
        preprocessed_las.predictions = predictions.numpy()
        preprocessed_las.add_extra_dim(laspy.ExtraBytesParams(name="error", type=np.int32))
        preprocessed_las.error = error.numpy()
        preprocessed_las.add_extra_dim(laspy.ExtraBytesParams(name="voxel_labels_0", type=np.int32))
        preprocessed_las.voxel_labels_0 = labels[:, 0].numpy()
        preprocessed_las.add_extra_dim(laspy.ExtraBytesParams(name="voxel_labels_1", type=np.int32))
        preprocessed_las.voxel_labels_1 = labels[:, 1].numpy()
        preprocessed_las.add_extra_dim(laspy.ExtraBytesParams(name="voxel_labels_2", type=np.int32))
        preprocessed_las.voxel_labels_2 = labels[:, 2].numpy()
        
        # Write to file
        preprocessed_las.write(preprocessed_path)

    # Create DataFrame and save to CSV
    if results_dfs:
        results_df = pd.concat(results_dfs)
        
        # Reorder columns for better readability
        column_order = [
            "filename",
            "point_count",
            "accuracy",
            "miou",
            "duration",
        ]
        results_df = results_df[column_order]
        results_df.to_csv(output_path, index=False)
        
        # Print summary
        print("\nResults Summary:")
        print(results_df.to_string())
        print(f"\nResults saved to: {output_path}")
    else:
        print("No results to save.")
