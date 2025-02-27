import torch
import time
import os
import laspy
import pandas as pd
from gaussian_mixture_cpp import hierarchical_gmm, GMMVariant
from pathlib import Path
import math

def batch_sqrt_lower_triangular(L):
    S = torch.zeros_like(L)
    S[:, 0, 0] = torch.sqrt(L[:, 0, 0])
    S[:, 1, 0] = L[:, 1, 0] / S[:, 0, 0]
    S[:, 1, 1] = torch.sqrt(L[:, 1, 1] - S[:, 1, 0]**2)
    S[:, 2, 0] = L[:, 2, 0] / S[:, 0, 0]
    S[:, 2, 1] = (L[:, 2, 1] - S[:, 1, 0] * S[:, 2, 0]) / S[:, 1, 1]
    S[:, 2, 2] = torch.sqrt(L[:, 2, 2] - S[:, 2, 0]**2 - S[:, 2, 1]**2)
    return S

def compute_accuracy(labels, y):
    accuracy = 0.0
    for label in labels.unique():
        most_common_label = torch.mode(y[labels == label])[0]
        accuracy += (y[labels == label] == most_common_label).float().mean()
    return accuracy / len(labels.unique())

def test_em(
    source_path, 
    hierarchy_k, 
    alpha, 
    tol, 
    max_iter, 
    variant,
    over_iter,
):
    """Process a LAS file with Hierarchical GMM and save results"""
    print(f"Loading data from {source_path}")
    
    # Read LAS file
    lasdata = laspy.read(source_path)
    # Load points and scale them
    x = torch.tensor(lasdata.xyz, dtype=torch.float32, device="cpu") * 10
    y = torch.tensor(lasdata.semantic_label.copy(), dtype=torch.int32, device="cpu")
    print("\nInput data info:")
    print(f" - Number of points: {x.shape[0]}")
    print(f" - Dimensions: {x.shape[1]}")
    print(f" - Device: {x.device}")
    # Convert hierarchy_k to tensor
    hierarchy_k_tensor = torch.tensor(hierarchy_k, dtype=torch.long, device=x.device)
    print(f"\nRunning Hierarchical GMM with variant: {variant}")
    
    results_dict = {
        "duration": [], 
        "accuracy": [],
        "iterations": [],
        "point_count": x.shape[0],
        "filename": os.path.basename(source_path)
    }
    
    for i in range(over_iter):
        start_time = time.time()
        all_level_labels, _, _, _ = hierarchical_gmm(
            x,
            hierarchy_k_tensor,
            alpha,
            tol,
            max_iter + i,
            variant
        )
        duration = time.time() - start_time
        print(f"Iteration {i+1}/{over_iter} completed in {duration:.3f} seconds")
        
        # Use the final level labels for accuracy computation
        final_labels = all_level_labels[-1]
        accuracy = compute_accuracy(final_labels, y)
        print(f"Accuracy: {accuracy:.3f}")
        
        results_dict["duration"].append(duration)
        results_dict["iterations"].append(max_iter + i)
        results_dict["accuracy"].append(accuracy)
        
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test Hierarchical GMM variants')
    parser.add_argument('--input', type=str, required=True, help='Input directory or path to LAS folder/file')
    parser.add_argument('--output', type=str, help='Output path for results CSV file')
    parser.add_argument('--variants', type=str, default="CEM", help='Variants to test')
    parser.add_argument('--approx_cluster_size', type=str, default="8192", help='Approximate cluster size')
    parser.add_argument('--depth', type=int, default=3, help='Depth of the hierarchy')
    parser.add_argument('--alpha', type=float, default=1e1, help='Alpha')
    parser.add_argument('--tol', type=float, default=1.0, help='Tolerance')
    parser.add_argument('--max_iter', type=int, default=1, help='Maximum iterations')
    parser.add_argument('--over_iter', type=int, default=5, help='Over iterations')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
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
    
    if Path(input).is_file():
        source_path = [input]
    elif Path(input).is_dir():
        source_path = os.listdir(input)
    else:
        raise ValueError(f"Invalid input path: {input}")
    
    if output is None:
        output_path = Path(source_path[0]).parent / "results.csv"
    elif Path(output).is_dir():
        output_path = output / "results.csv"
    elif Path(output).is_file():
        output_path = output
    else:
        raise ValueError(f"Invalid output path: {output}")
    
    variants = variants.split(",")
    variants = [getattr(GMMVariant, variant) for variant in variants]
    
    # Generate hierarchy_ks using the new function
    hierarchy_ks = [make_hierarchy(int(size), depth) for size in approx_cluster_size.split(",")]
    
    results_dfs = []
    
    for variant in variants:
        for hierarchy_k in hierarchy_ks:
            actual_size = math.prod(hierarchy_k)
            for file in source_path:
                print(f"\nTesting {variant} on {file} with hierarchy_k: {hierarchy_k} ({actual_size}):")
                result_dict = test_em(
                    source_path=os.path.join(input, file),
                    hierarchy_k=hierarchy_k,
                    alpha=alpha,
                    tol=tol,
                    max_iter=max_iter,
                    variant=variant,
                    over_iter=over_iter,
                )
                # Add variant information to result dictionary
                result_dict["variant"] = variant.name
                result_dict["actual_size"] = actual_size
                result_dict["depth"] = depth
                results_dfs.append(pd.DataFrame(result_dict))

    # Create DataFrame and save to CSV
    results_df = pd.concat(results_dfs)
    
    # Reorder columns for better readability
    column_order = ['filename', 'variant', 'point_count', 'accuracy', 'duration', 'iterations', 'actual_size', 'depth']
    results_df = results_df[column_order]
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\nResults Summary:")
    print(results_df.to_string())
    print(f"\nResults saved to: {output_path}")