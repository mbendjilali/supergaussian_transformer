import os
import glob
import numpy as np
import pandas as pd
import laspy
import open3d as o3d
from pathlib import Path
import logging
from typing import Optional, Tuple, List

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# S3DIS semantic labels mapping
OBJECT_LABEL = {
    'ceiling': 0,
    'floor': 1,
    'wall': 2,
    'beam': 3,
    'column': 4,
    'window': 5,
    'door': 6,
    'table': 7,
    'chair': 8,
    'sofa': 9,
    'bookcase': 10,
    'board': 11,
    'clutter': 12
}

def grid_subsample(
    points: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    grid_size: float,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subsample point cloud using a regular grid.
    
    Args:
        points: Nx3 array of point coordinates
        rgb: Nx3 array of RGB values
        labels: N array of semantic labels
        grid_size: Size of the grid cells for subsampling
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - subsampled_points: Kx3 array of subsampled point coordinates
        - subsampled_rgb: Kx3 array of subsampled RGB values
        - subsampled_labels: K array of subsampled semantic labels
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Normalize and set RGB colors
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32) / 255.0)
    
    # Voxel downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=grid_size)
    
    # Extract downsampled points and colors
    points_down = np.asarray(pcd_down.points)
    rgb_down = np.asarray(pcd_down.colors) * 255.0
    
    # For labels, we'll use nearest neighbor interpolation
    # by finding the closest original point to each downsampled point
    tree = o3d.geometry.KDTreeFlann(pcd)
    labels_down = []
    
    for point in points_down:
        _, idx, _ = tree.search_knn_vector_3d(point, 1)
        labels_down.append(labels[idx[0]])
    
    labels_down = np.array(labels_down, dtype=np.int32)
    
    if verbose:
        reduction = 100 * (1 - len(points_down) / len(points))
        log.info(f"Grid sampling reduced points by {reduction:.1f}% "
                f"({len(points)} -> {len(points_down)} points)")
    
    return points_down, rgb_down.astype(np.uint8), labels_down

def read_room_to_numpy(
    room_dir: str,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read S3DIS room data from numpy files.

    Args:
        room_dir: Path to the room directory containing .npy files
        verbose: Whether to print debug information

    Returns:
        Tuple containing:
        - xyz: Nx3 array of point coordinates
        - rgb: Nx3 array of RGB colors
        - labels: Nx1 array of semantic labels
        
    Raises:
        ValueError: If required files are not found or invalid
    """
    if verbose:
        log.info(f"Reading room: {room_dir}")

    # Check for required files
    coord_path = os.path.join(room_dir, 'coord.npy')
    color_path = os.path.join(room_dir, 'color.npy')
    segment_path = os.path.join(room_dir, 'segment.npy')

    if not all(os.path.exists(p) for p in [coord_path, color_path, segment_path]):
        raise ValueError(f"Missing required .npy files in: {room_dir}")

    try:
        # Load coordinates, colors and semantic labels
        xyz = np.load(coord_path).astype(np.float32)  # Nx3
        rgb = np.load(color_path).astype(np.uint8)    # Nx3
        labels = np.load(segment_path).astype(np.int32)  # Nx1

        # Squeeze labels from (N,1) to (N,)
        labels = np.squeeze(labels)

        if xyz.shape[0] == 0:
            raise ValueError("Empty point cloud")
            
        if not (xyz.shape[0] == rgb.shape[0] == labels.shape[0]):
            raise ValueError("Inconsistent number of points across files")

        if verbose:
            log.info(f"Successfully read {xyz.shape[0]} points")

        return xyz, rgb, labels

    except Exception as e:
        raise ValueError(f"Error reading data from {room_dir}: {str(e)}")

def convert_area_to_merged_las(
    area_dir: str,
    output_path: str,
    grid_size: float = 0.03,
    verbose: bool = False
) -> None:
    """Convert all rooms in an S3DIS area to a single merged LAS file.

    Args:
        area_dir: Path to the area directory
        output_path: Path where to save the merged LAS file
        grid_size: Size of the grid cells for subsampling (in meters)
        verbose: Whether to print debug information
    """
    # List all room directories (they should contain .npy files)
    room_dirs = sorted([
        d for d in glob.glob(os.path.join(area_dir, '*'))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'coord.npy'))
    ])

    if not room_dirs:
        raise ValueError(f"No valid room directories found in: {area_dir}")

    # Initialize lists to store all points
    all_xyz = []
    all_rgb = []
    all_labels = []

    # Read all rooms
    for i, room_dir in enumerate(room_dirs):
        if verbose:
            log.info(f"Processing room {i+1}/{len(room_dirs)}: {room_dir}")
            
        try:
            xyz, rgb, labels = read_room_to_numpy(room_dir, verbose)
            
            if grid_size > 0:
                # Grid subsample the point cloud
                xyz_down, rgb_down, labels_down = grid_subsample(
                    xyz, rgb, labels, grid_size, verbose)
            else:
                xyz_down = xyz
                rgb_down = rgb
                labels_down = labels
            
            # Store the subsampled data
            all_xyz.append(xyz_down)
            all_rgb.append(rgb_down)
            all_labels.append(labels_down)
            
        except Exception as e:
            log.error(f"Failed to process room {room_dir}: {str(e)}")
            continue

    # Concatenate all points
    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if verbose:
        log.info(f"Total points in merged cloud: {len(xyz)}")

    try:
        # Create LAS file
        header = laspy.LasHeader(point_format=2, version="1.2")
        
        # Add custom dimensions for semantic labels and room IDs
        header.add_extra_dim(laspy.ExtraBytesParams(
            name="semantic_label",
            type=np.int32,
            description="Semantic label"
        ))

        las = laspy.LasData(header)
        
        # Set coordinates
        las.x = xyz[:, 0]
        las.y = xyz[:, 1]
        las.z = xyz[:, 2]

        # Set colors (scale from 0-255 to 0-65535 as required by LAS format)
        las.red = rgb[:, 0].astype(np.uint16) * 256
        las.green = rgb[:, 1].astype(np.uint16) * 256
        las.blue = rgb[:, 2].astype(np.uint16) * 256

        # Set semantic labels and room IDs
        las.semantic_label = labels

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save merged LAS file
        las.write(output_path)
        if verbose:
            log.info(f"Saved merged point cloud to: {output_path}")
            
    except Exception as e:
        log.error(f"Failed to save merged point cloud: {str(e)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert S3DIS dataset to merged LAS format')
    parser.add_argument('--input', type=str, required=True, help='Path to S3DIS raw directory')
    parser.add_argument('--output', type=str, required=True, help='Output path for merged LAS file')
    parser.add_argument('--area', type=str, default='Area_5', help='Area to convert (e.g. Area_5)')
    parser.add_argument('--grid', type=float, default=0.03, help='Grid size for subsampling in meters')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    args = parser.parse_args()

    area_dir = os.path.join(args.input, args.area)
    if not os.path.exists(area_dir):
        raise ValueError(f"Area directory not found: {area_dir}")

    # Construct output path
    output_path = args.output
    if not output_path.endswith('.las'):
        output_path = os.path.join(output_path, f"{args.area}.las")

    convert_area_to_merged_las(area_dir, output_path, args.grid, args.verbose)

if __name__ == '__main__':
    main() 