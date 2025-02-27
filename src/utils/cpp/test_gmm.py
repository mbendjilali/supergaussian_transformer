import torch
import time
import laspy
import numpy as np
from gaussian_mixture_cpp import bayesian_gaussian_mixture_model

def process_las_with_gmm(source_path, output_path, n_components=16, alpha=1.0, tol=1e-2, max_iter=10, subsample_ratio=0.01):
    """Process a LAS file with Gaussian Mixture Model and save results"""
    print(f"Loading data from {source_path}")
    
    # Read LAS file
    lasdata = laspy.read(source_path)
    
    try:
        # Create a copy for output
        output_las = laspy.LasData(lasdata.header)
        
        # Load points and scale them
        x = torch.tensor(lasdata.xyz, dtype=torch.float32, device="cuda") * 10
        
        print("\nInput data info:")
        print(f" - Number of points: {x.shape[0]}")
        print(f" - Dimensions: {x.shape[1]}")
        print(f" - Device: {x.device}")
        
        # Run GMM
        print(f"\nRunning GMM with {n_components} components and {subsample_ratio:.1%} subsample ratio...")
        start_time = time.time()
        
        pi, mu, sigma, labels = bayesian_gaussian_mixture_model(
            x,
            n_components,
            alpha,
            tol,
            max_iter
        )
        
        duration = time.time() - start_time
        print(f"GMM completed in {duration:.3f} seconds")
        
        # Copy all data from input to output
        for dimension in lasdata.point_format.dimension_names:
            setattr(output_las, dimension, getattr(lasdata, dimension))
        
        # Store cluster labels in point_source_id instead of classification
        output_las.point_source_id = labels.cpu().numpy().astype(np.uint16)
        
        # Save result
        print(f"\nSaving results to {output_path}")
        output_las.write(output_path)
        
        print("Done!")
        
    except Exception as e:
        print(f"Error during GMM: {str(e)}")
        return


if __name__ == "__main__":
    source_path = "/home/moussabendjilali/toy scene.las"
    output_path = "/home/moussabendjilali/toy_scene_bgmm_result.las"
    
    process_las_with_gmm(
        source_path=source_path,
        output_path=output_path,
        n_components=512,
        alpha=1.0,
        tol=1e-2,
        max_iter=10,
        subsample_ratio=0.1
    )