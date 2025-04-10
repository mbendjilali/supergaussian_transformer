import argparse
import time
from pathlib import Path
import torch
import laspy
import pandas as pd
import numpy as np
from gaussian_mixture_cpp import GMMVariant

# Import from test_hgmm.py
from test_hgmm import (
    test_em,
    baseline as grid_baseline,
    make_hierarchy,
    downsample_point_cloud
)

# Import from test_cutpursuit.py
from test_cutpursuit import test_cutpursuit

def calculate_snr_db(signal, noise):
    """Calculate Signal-to-Noise Ratio in decibels."""
    signal_power = torch.mean(signal ** 2)
    noise_power = torch.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * torch.log10(signal_power / noise_power).item()

def add_gaussian_noise(x, noise_level):
    """Add Gaussian noise to point cloud coordinates."""
    if noise_level == 0:
        return x, float('inf')
    noise = torch.randn_like(x) * noise_level
    snr_db = calculate_snr_db(x, noise)
    return x + noise, snr_db

def run_comparison(
    input_path,
    output_path,
    noise_levels=[0.0],
    num_repetitions=1,
    gmm_variants="",
    cutpursuit=False,
    baseline=False,
    gmm_cluster_sizes=[8192],
    gmm_depth=4,
    gmm_alpha=1.0,
    gmm_tol=1e-2,
    gmm_max_iter=5,
    cutpursuit_reg=[30.0],
    cutpursuit_sw=[1.0],
    cutpursuit_cutoff=[10],
    grid_sizes=[0.1],
    downsample_size=None,
    k_neighbors=50,
    verbose=True
):
    """Run comprehensive comparison of segmentation algorithms."""
    
    if not cutpursuit and not baseline and gmm_variants == "":
        raise ValueError("At least one of cutpursuit, baseline, or gmm_variants must be provided")
    
    # Process input path
    if Path(input_path).is_file():
        source_path = [input_path]
    elif Path(input_path).is_dir():
        source_path = [file for file in Path(input_path).iterdir() 
                      if file.is_file() and file.suffix == ".las"]
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    results_dfs = []

    # Process each file
    for file in source_path:
        print(f"\nProcessing {file}")
        lasdata = laspy.read(file)
        
        # Load or downsample point cloud
        if downsample_size is not None:
            x, y, rgb = downsample_point_cloud(lasdata, downsample_size)
            print(f"Downsampled to {x.shape[0]} points")
        else:
            x = torch.tensor(lasdata.xyz, dtype=torch.float32)
            y = torch.tensor(lasdata.sem_class.copy(), dtype=torch.int32)
            rgb = torch.tensor(np.vstack([lasdata.red, lasdata.green, lasdata.blue]).T / 65535.0, 
                             dtype=torch.float32)

        # Test with different noise levels
        for noise_level in noise_levels:
            print(f"\nTesting with noise level: {noise_level}")
            
            # Repeat experiment N times
            for rep in range(num_repetitions):
                print(f"\nRepetition {rep + 1}/{num_repetitions}")
                
                # Add noise to coordinates
                noisy_x, snr_db = add_gaussian_noise(x, noise_level)
                
                # 1. Test GMM variants
                if gmm_variants != "":
                    for variant_name in gmm_variants:
                        variant = getattr(GMMVariant, variant_name)
                        for cluster_size in gmm_cluster_sizes:
                            hierarchy_k = make_hierarchy(cluster_size, gmm_depth)
                            actual_size = np.prod(hierarchy_k)
                            
                            print(f"\nTesting {variant_name}")
                            result_dict, predictions, error, cluster, mu, sigma = test_em(
                                x=noisy_x,
                                y=y,
                                hierarchy_k=hierarchy_k,
                                alpha=gmm_alpha,
                                tol=gmm_tol,
                                max_iter=gmm_max_iter,
                                variant=variant,
                            )
                            # Draw 100,000 points from the Gaussians (mu, sigma) and save to LAS file
                            if mu is not None and sigma is not None:
                                # Get the last level of the hierarchy
                                last_level_mu = mu[-1]  # Shape: [K, D]
                                last_level_sigma = sigma[-1]  # Shape: [K, D, D]
                                
                                # Number of points to sample from each Gaussian
                                num_gaussians = last_level_mu.shape[0]
                                points_per_gaussian = 100000 // num_gaussians
                                
                                # Sample points from each Gaussian
                                sampled_points = []
                                for i in range(num_gaussians):
                                    # Create multivariate normal distribution
                                    mvn = torch.distributions.MultivariateNormal(
                                        loc=last_level_mu[i],
                                        covariance_matrix=last_level_sigma[i]
                                    )
                                    
                                    # Sample points
                                    samples = mvn.sample((points_per_gaussian,))
                                    sampled_points.append(samples)
                                
                                # Combine all sampled points
                                all_samples = torch.cat(sampled_points, dim=0)
                                
                                # Create LAS file
                                fil = f"{Path(file).stem}_gmm_{variant_name}_size_{actual_size}.las"
                                ou = Path(file).parent / fil
                                
                                # Create LAS header
                                header = laspy.LasHeader(version="1.4")
                                header.offsets = np.min(all_samples.cpu().numpy(), axis=0)
                                header.scales = np.array([0.001, 0.001, 0.001])
                                
                                # Create LAS data
                                las = laspy.LasData(header)
                                las.x = all_samples[:, 0].cpu().numpy()
                                las.y = all_samples[:, 1].cpu().numpy()
                                las.z = all_samples[:, 2].cpu().numpy()
                                # Save LAS file
                                las.write(ou)
                                print(f"Saved {all_samples.shape[0]} sampled points to {ou}")
                            # Add metadata to results
                            row = {
                                "filename": str(file),
                                "point_count": x.shape[0],
                                "snr_db": snr_db,
                                "noise_level": noise_level,
                                "repetition": rep + 1,
                                "algorithm": f"GMM_{variant_name}",
                                "num_components": cluster[-1].unique().shape[0],
                                "depth": gmm_depth,
                                "accuracy": result_dict["accuracy"],
                                "miou": result_dict["miou"],
                                "duration": result_dict["duration"],
                                "iterations": result_dict["iterations"]
                            }
                            results_dfs.append(pd.DataFrame([row]))
                
                # 2. Test CutPursuit
                if cutpursuit:
                    for reg, sw, cut in zip(cutpursuit_reg, cutpursuit_sw, cutpursuit_cutoff):
                        print(f"\nTesting CutPursuit")
                        result_dict = test_cutpursuit(
                            filename=file,
                            x=noisy_x,
                            y=y,
                            rgb=rgb,
                            regularization=reg,
                            spatial_weight=sw,
                            cutoff=cut,
                            iterations=10,
                            r_max=1.0,
                            k_neighbors=k_neighbors,
                            parallel=True,
                            verbose=verbose
                        )
                    
                        row = {
                            "filename": str(file),
                            "point_count": x.shape[0],
                            "snr_db": snr_db,
                            "noise_level": noise_level,
                            "repetition": rep + 1,
                            "algorithm": "CutPursuit",
                            "accuracy": result_dict["accuracy"],
                            "miou": result_dict["miou"],
                            "duration": result_dict["duration"],
                            "iterations": result_dict["iterations"],
                            "num_components": result_dict["num_components"],
                            "depth": 1
                        }
                        results_dfs.append(pd.DataFrame([row]))
                    
                # 3. Test Grid baseline
                if baseline:
                    for grid_size in grid_sizes:
                        print(f"\nTesting Grid baseline")
                        result_dict = grid_baseline(noisy_x, y, grid_size, file)
                        row = {
                            "filename": str(file),
                            "point_count": x.shape[0],
                            "snr_db": snr_db,
                            "noise_level": noise_level,
                            "repetition": rep + 1,
                            "algorithm": "Grid",
                            "accuracy": result_dict["accuracy"],
                            "miou": result_dict["miou"],
                            "duration": result_dict["duration"],
                            "iterations": result_dict["iterations"],
                            "num_components": result_dict["actual_size"],
                            "depth": 1
                        }
                        print("Duration: ", result_dict["duration"])
                        print("Mean class accuracy: ", result_dict["accuracy"])
                        print("Mean IoU: ", result_dict["miou"])
                        results_dfs.append(pd.DataFrame([row]))

    # Combine all results
    results_df = pd.concat(results_dfs, ignore_index=True)
    
    # Reorder columns
    column_order = [
        "filename",
        "point_count",
        "snr_db",
        "noise_level",
        "repetition",
        "accuracy",
        "miou",
        "duration",
        "iterations",
        "num_components",
        "depth",
        "algorithm"
    ]
    results_df = results_df[column_order]
    
    # Compute averaged results
    metrics_to_average = ['accuracy', 'miou', 'duration', 'iterations']
    groupby_columns = ['filename', 'algorithm', 'noise_level', 'num_components', 'depth']
    
    # Calculate mean and standard deviation
    avg_results = results_df.groupby(groupby_columns)[metrics_to_average].agg(['mean', 'std']).reset_index()
    
    # Flatten multi-level columns
    avg_results.columns = [
        col[0] if col[1] == '' else f"{col[0]}_{col[1]}" 
        for col in avg_results.columns
    ]
    
    # Add point_count and snr_db (they're constant per group)
    avg_results = avg_results.merge(
        results_df.groupby(groupby_columns)[['point_count', 'snr_db']].first().reset_index(),
        on=groupby_columns
    )
    
    # Reorder columns for averaged results
    avg_column_order = [
        "filename",
        "point_count",
        "snr_db",
        "noise_level",
        "accuracy_mean",
        "accuracy_std",
        "miou_mean",
        "miou_std",
        "duration_mean",
        "duration_std",
        "iterations_mean",
        "iterations_std",
        "num_components",
        "depth",
        "algorithm"
    ]
    avg_results = avg_results[avg_column_order]
    
    # Save both detailed and averaged results
    results_df.to_csv(output_path, index=False)
    avg_output_path = str(Path(output_path).with_suffix('')) + '_averaged.csv'
    avg_results.to_csv(avg_output_path, index=False)
    
    print(f"\nDetailed results saved to: {output_path}")
    print(f"Averaged results saved to: {avg_output_path}")
    print("\nDetailed Results Summary:")
    print(results_df.to_string())
    print("\nAveraged Results Summary:")
    print(avg_results.to_string())
    
    return results_df, avg_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare segmentation algorithms")
    
    # Input/output arguments
    parser.add_argument("--input", required=True, type=str,
                      help="Input directory or path to LAS folder/file")
    parser.add_argument("--output", type=str, default=None,
                      help="Output path for results CSV file")
    
    # Experiment parameters
    parser.add_argument("--noise_levels", type=str, default="0.0",
                      help="Comma-separated list of noise standard deviations")
    parser.add_argument("--num_repetitions", type=int, default=1,
                      help="Number of times to repeat each experiment")
    parser.add_argument("--gmm_variants", type=str,
                      help="Comma-separated list of GMM variants to test")
    parser.add_argument("--cut_pursuit", action="store_true",
                      help="Run CutPursuit")
    parser.add_argument("--grid_baseline", action="store_true",
                      help="Run Grid baseline")
    
    # GMM parameters
    parser.add_argument("--gmm_cluster_sizes", type=str, default="8192",
                      help="Comma-separated list of approximate cluster sizes")
    parser.add_argument("--gmm_depth", type=int, default=3,
                      help="Depth of GMM hierarchy")
    parser.add_argument("--gmm_alpha", type=float, default=1.0,
                      help="GMM regularization parameter")
    parser.add_argument("--gmm_tol", type=float, default=1e-2,
                      help="GMM convergence tolerance")
    parser.add_argument("--gmm_max_iter", type=int, default=10,
                      help="Maximum GMM iterations")
    
    # CutPursuit parameters
    parser.add_argument("--cutpursuit_reg", type=str, default="1.0",
                      help="Comma-separated list of regularization parameters")
    parser.add_argument("--cutpursuit_sw", type=str, default="1.0",
                      help="Comma-separated list of spatial weights")
    parser.add_argument("--cutpursuit_cutoff", type=str, default="5",
                      help="Comma-separated list of cutoff values")
    
    # Grid baseline parameters
    parser.add_argument("--grid_sizes", type=str, default="0.1",
                      help="Comma-separated list of grid sizes for baseline")
    
    # General parameters
    parser.add_argument("--downsample_size", type=float, default=None,
                      help="Grid size for downsampling input points")
    parser.add_argument("--k_neighbors", type=int, default=50,
                      help="Number of neighbors for feature computation")
    parser.add_argument("--verbose", action="store_true",
                      help="Print debug information")
    
    args = parser.parse_args()
    
    # Parse lists from command line arguments
    noise_levels = [float(x) for x in args.noise_levels.split(",")]
    gmm_variants = args.gmm_variants.split(",") if args.gmm_variants is not None else []
    cutpursuit = args.cut_pursuit
    baseline = args.grid_baseline
    gmm_cluster_sizes = [int(x) for x in args.gmm_cluster_sizes.split(",")]
    cutpursuit_reg = [float(x) for x in args.cutpursuit_reg.split(",")]
    cutpursuit_sw = [float(x) for x in args.cutpursuit_sw.split(",")]
    cutpursuit_cutoff = [int(x) for x in args.cutpursuit_cutoff.split(",")]
    grid_sizes = [float(x) for x in args.grid_sizes.split(",")]
    
    # Set default output path if not provided
    if args.output is None:
        output_path = Path(args.input).parent / "segmentation_comparison.csv"
    else:
        output_path = args.output
    
    # Run comparison
    run_comparison(
        input_path=args.input,
        output_path=output_path,
        noise_levels=noise_levels,
        num_repetitions=args.num_repetitions,
        gmm_variants=gmm_variants,
        cutpursuit=cutpursuit,
        baseline=baseline,
        gmm_cluster_sizes=gmm_cluster_sizes,
        gmm_depth=args.gmm_depth,
        gmm_alpha=args.gmm_alpha,
        gmm_tol=args.gmm_tol,
        gmm_max_iter=args.gmm_max_iter,
        cutpursuit_reg=cutpursuit_reg,
        cutpursuit_sw=cutpursuit_sw,
        cutpursuit_cutoff=cutpursuit_cutoff,
        grid_sizes=grid_sizes,
        downsample_size=args.downsample_size,
        k_neighbors=args.k_neighbors,
        verbose=args.verbose
    ) 