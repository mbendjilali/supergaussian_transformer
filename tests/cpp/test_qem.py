import os.path as osp
import sys
import time
from pathlib import Path

import laspy
import torch
import numpy as np

current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(osp.dirname(current_dir))
sys.path.append(project_root)

import qem_cpp
from src.utils.neighbors import knn_1_graph
from src.transforms.point import PointFeatures
from src.data import Data


def test_qem(file_path, k_init=10, max_iter=2, reg=0.01, k_neighbor=20, qem_tol=1e-4):
    """
    Test the QEM region growing algorithm on a point cloud.

    Args:
        file_path: Path to a LAS point cloud file
        k_init: Initial number of clusters
        reg: Regularization parameter
        k_neighbor: Number of neighbors for building the graph

    Returns:
        Result of QEM computation
    """
    # Load point cloud
    print(f"Loading point cloud from {file_path}")
    lasdata = laspy.read(file_path)
    points = torch.tensor(lasdata.xyz, dtype=torch.float32)

    # Precomputation
    print(f"Running KNN with k_neighbor={k_neighbor}")
    start_time = time.time()
    neighbors, distances = knn_1_graph(
        points,
        k=k_neighbor,
        r_max=float("inf"),
        batch=None,
        oversample=True,
        self_is_neighbor=False,
        verbose=False,
        trim=False,
    )
    neighbors = neighbors[1].view(-1, k_neighbor)
    distances = distances.view(-1, k_neighbor)

    true_reg = 4.0 * reg * k_init * torch.sqrt(distances.mean()).item()

    data = Data(
        pos=points,
        neighbor_index=neighbors,
        neighbor_distance=distances,
    )
    print(f"Estimating normal vectors")
    point_features = PointFeatures(keys=["normal"], k_min=5, k_step=-1, overwrite=True)
    data = point_features(data)
    normals = data.normal
    qem_time = time.time() - start_time
    print(f"Precomputation completed in {qem_time:.3f} seconds")

    # Run QEM
    print(f"Running QEM with k_init={k_init}, reg={true_reg}")
    start_time = time.time()
    generators, assignments = qem_cpp.qem_partitioning(
        data.pos,
        data.neighbor_index,
        data.neighbor_distance,
        data.normal,
        k_init,
        true_reg,
        qem_tol,
        max_iter,
    )
    qem_time = time.time() - start_time
    print(f"QEM computation completed in {qem_time:.3f} seconds")

    return neighbors, distances, normals, generators, assignments


def argument_parser():
    """Create argument parser for the QEM testing script."""
    import argparse

    parser = argparse.ArgumentParser(description="Test QEM region growing algorithm")

    # Input arguments
    parser.add_argument(
        "--input", default="/home/moussabendjilali/fake_scene_1.las", type=str, help="Input path to LAS file"
    )

    # QEM parameters
    parser.add_argument(
        "--k_init", default=10, type=int, help="Initial number of clusters"
    )
    parser.add_argument(
        "--max_iter", default=2, type=int, help="Maximum number of iter"
    )
    parser.add_argument(
        "--reg", default=0.0, type=float, help="Regularization parameter"
    )
    parser.add_argument(
        "--qem_tol", default=1e-4, type=float, help="QEM tolerance"
    )
    parser.add_argument(
        "--k_neighbor",
        default=20,
        type=int,
        help="Number of neighbors for graph construction",
    )

    return parser


if __name__ == "__main__":
    # Parse command line arguments
    parser = argument_parser()
    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists() or input_path.suffix.lower() != ".las":
        raise ValueError(f"Input must be a valid LAS file: {args.input}")

    # Run QEM test
    neighbors, distances, normals, generators, assignments = test_qem(
        file_path=args.input,
        k_init=args.k_init,
        max_iter=args.max_iter,
        reg=args.reg,
        qem_tol=args.qem_tol,
        k_neighbor=args.k_neighbor,
    )

    # Save assignments to new LAS file
    output_path = input_path.parent / f"{input_path.stem}_qem.las"
    
    # Read original LAS file to copy header and point format
    las = laspy.read(input_path)
    
    # Create new LAS with same header/format
    new_las = laspy.LasData(las.header)
    new_las.points = las.points
    
    # Add assignments as a new field
    new_las.add_extra_dim(laspy.ExtraBytesParams(name="qem_assignment", type=np.int32))
    new_las.qem_assignment = assignments.numpy()
    
    # Write output file
    new_las.write(output_path)
    print(f"Saved QEM assignments to {output_path}")
    print("Test completed successfully")
