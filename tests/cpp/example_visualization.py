#!/usr/bin/env python3
"""
Example script demonstrating hGMM visualization with GEM variant.

This script shows how to run the hierarchical Gaussian Mixture Model with GEM variant
and generate an interactive HTML visualization showing the point cloud and Gaussians
at each partition level.

Usage:
    python example_visualization.py --input path/to/pointcloud.las --output visualization.html
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Example usage of the visualization script
    print("hGMM Visualization Example")
    print("=" * 50)
    
    # Check if input file is provided
    if len(sys.argv) < 2:
        print("Usage: python example_visualization.py <input_las_file> [output_html_file]")
        print("\nExample:")
        print("  python example_visualization.py data/pointcloud.las")
        print("  python example_visualization.py data/pointcloud.las my_visualization.html")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    # Build command
    cmd = [
        "python", "test_hgmm_visualization.py",
        "--input", input_file,
        "--cluster_size", "32",  # Smaller for faster processing
        "--depth", "3",          # 3 levels for visualization
        "--max_iter", "2",       # Fewer iterations for demo
        "--alpha", "1.0",
        "--tol", "0.01",
        "--sample_rate", "50",   # Sample every 50th point for smaller file
        "--downsample_grid_size", "1.0"  # Downsample for smaller file
    ]
    
    if output_file:
        cmd.extend(["--output", output_file])
    
    print(f"Running command: {' '.join(cmd)}")
    print("\nThis will:")
    print("1. Load the point cloud from the LAS file")
    print("2. Run hGMM with GEM variant")
    print("3. Generate an interactive HTML visualization")
    print("4. Show point cloud colored by clusters and Gaussian ellipsoids at each level")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("\n" + result.stdout)
        print("Visualization completed successfully!")
        
        if output_file:
            print(f"Open '{output_file}' in your web browser to view the visualization.")
        else:
            # Find the generated file
            input_path = Path(input_file)
            default_output = input_path.parent / f"hgmm_visualization_{input_path.stem}.html"
            if default_output.exists():
                print(f"Open '{default_output}' in your web browser to view the visualization.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running visualization: {e}")
        print(f"Error output: {e.stderr}")
    except FileNotFoundError:
        print("Error: Could not find test_hgmm_visualization.py")
        print("Make sure you're running this from the tests/cpp directory.")

if __name__ == "__main__":
    main() 