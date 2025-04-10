import torch
import time
import numpy as np
import laspy
import pandas as pd
from pathlib import Path
import sys
import os.path as osp

# Add necessary paths for imports
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(osp.dirname(current_dir))  # Go up 2 levels to reach project root
sys.path.append(project_root)  # Add project root to Python path

# Now add dependencies paths
dependencies_folder = osp.join(project_root, "src/dependencies")
sys.path.append(osp.join(dependencies_folder, "grid_graph/python"))
sys.path.append(osp.join(dependencies_folder, "parallel_cut_pursuit/python"))

# After adding paths, import the required modules
from grid_graph import edge_list_to_forward_star
from src.dependencies.parallel_cut_pursuit.python.wrappers.cp_d0_dist import cp_d0_dist
from src.utils.cpu import available_cpu_count
from torch_scatter import scatter_sum
from pandarallel import pandarallel
from src.utils.neighbors import knn_1_graph
from src.transforms.point import PointFeatures, GroundElevation
from src.data import Data

pandarallel.initialize()

def hard_assign(labels, y):
    # Convert to pandas series
    labels_series = pd.Series(labels.cpu().numpy())
    y_series = pd.Series(y.cpu().numpy())
    predictions = labels_series.groupby(labels_series).transform(lambda x: y_series[x.index].mode()[0])  
    # Convert back to torch tensor on same device as input
    return torch.tensor(predictions.values, device=labels.device)

def compute_mean_accuracy(predictions, y):
    """Compute mean class accuracy for semantic segmentation."""
    unique_classes = y.unique()
    accuracies = []
    for cls in unique_classes:
        mask = (y == cls)
        if mask.sum() == 0:
            continue
            
        class_acc = (predictions[mask] == y[mask]).float().mean()
        accuracies.append(class_acc)
    
    return torch.stack(accuracies).mean().item()

def compute_miou(predictions, y, num_classes):
    """Compute mean IoU for semantic segmentation."""
    intersections = torch.zeros(num_classes, device=y.device)
    unions = torch.zeros(num_classes, device=y.device)
    
    for cls in range(num_classes):
        pred_mask = (predictions == cls)
        true_mask = (y == cls)
        intersection = (pred_mask & true_mask).sum().float()
        union = (pred_mask | true_mask).sum().float()
        
        intersections[cls] = intersection
        unions[cls] = union
    valid_classes = unions > 0
    if valid_classes.sum() == 0:
        return 0.0
        
    ious = intersections[valid_classes] / unions[valid_classes]
    return ious.mean().item()

def test_cutpursuit(
    filename,
    x,
    y,
    rgb,
    regularization,
    spatial_weight,
    cutoff,
    iterations,
    r_max=1.0,
    k_neighbors=50,  # Number of neighbors for geometric features
    k_adjacency=5,   # For connecting isolated nodes
    parallel=True,
    verbose=False
):
    """Test CutPursuit partition on point cloud data with hierarchical partitioning."""
    # Convert parameters to lists if they're not already
    regularization = [regularization] if not isinstance(regularization, list) else regularization
    spatial_weight = [spatial_weight] * len(regularization) if isinstance(spatial_weight, (float, int)) else spatial_weight
    cutoff = [cutoff] * len(regularization) if isinstance(cutoff, int) else cutoff
    
    assert len(regularization) == len(cutoff) == len(spatial_weight), "Parameter lists must have same length"
    
    # Initialize results dictionary with single-item lists for consistency
    results_dict = {
        "duration": 0.0,
        "accuracy": 0.0,
        "miou": 0.0,
        "iterations": 0,
        "point_count": 0,
        "filename": "",
        "regularization": 0.0,
        "spatial_weight": 0.0,
        "cutoff": 0,
        "r_max": 0.0,
        "k_neighbors": 0,
        "hierarchy_depth": 0,
        "num_components": 0
    }

    start_time = time.time()

    # First, get the k-nearest neighbors for feature computation
    neighbors, distances = knn_1_graph(
        x,
        k=k_neighbors,
        r_max=float('inf'),
        batch=None,
        oversample=True,
        self_is_neighbor=False,
        verbose=verbose,
        trim=False
    )

    # Create initial Data object
    data = Data(
        pos=x,
        rgb=rgb,
        neighbor_index=neighbors[1].view(-1, k_neighbors),
        neighbor_distance=distances.view(-1, k_neighbors)
    )

    # Compute geometric features
    point_features = PointFeatures(
        keys=['linearity', 'planarity', 'scattering', 'verticality'],
        k_min=5,
        k_step=-1,
        overwrite=True
    )
    data = point_features(data)

    # Compute elevation
    ground_elevation = GroundElevation(
        z_threshold=1.5,
        xy_grid=None,
        model='ransac',
        scale=4.0
    )
    data = ground_elevation(data)

    # Initialize data list for hierarchical partitioning
    data_list = [data]
    num_threads = available_cpu_count() if parallel else 1

    # Iteratively run the partition on the previous partition level
    for level, (reg, cut, sw) in enumerate(zip(regularization, cutoff, spatial_weight)):

        # Recover the Data object on which we will run the partition
        d1 = data_list[level]

        # Exit if the graph contains only one node
        if d1.num_nodes < 2:
            break

        # Create features for current level
        features = torch.cat([
            d1.rgb,
            d1.linearity,
            d1.planarity, 
            d1.scattering,
            d1.verticality,
            d1.elevation
        ], dim=1)

        # Create graph for current level
        edge_index, distances = knn_1_graph(
            d1.pos,
            k=10,
            r_max=r_max,
            batch=None,
            oversample=False,
            self_is_neighbor=False,
            verbose=verbose,
            trim=True
        )

        # Scale edge weights
        edge_weights = distances * reg

        # Convert to forward-star representation
        source_csr, target, reindex = edge_list_to_forward_star(
            d1.num_nodes, edge_index.T.contiguous().cpu().numpy())
        source_csr = source_csr.astype('uint32')
        target = target.astype('uint32')
        edge_weights = edge_weights.cpu().numpy()[reindex]

        # Prepare features
        n_dim = features.shape[1]
        pos_offset = features.mean(dim=0)
        features = features - pos_offset
        features = np.asfortranarray(features.cpu().numpy().T)
        
        # Set coordinate weights
        coor_weights = np.ones(n_dim, dtype=np.float32)
        coor_weights *= sw

        # Run CutPursuit
        super_index, x_c, cluster, edges, times = cp_d0_dist(
            n_dim,
            features,
            source_csr,
            target,
            edge_weights=edge_weights,
            vert_weights=d1.node_size.float().cpu().numpy() if hasattr(d1, 'node_size') else np.ones(d1.num_nodes, dtype=np.float32),
            coor_weights=coor_weights,
            min_comp_weight=cut,
            cp_dif_tol=1e-2,
            cp_it_max=iterations,
            split_damp_ratio=0.7,
            verbose=False,
            max_num_threads=num_threads,
            balance_parallel_split=True,
            compute_Time=True,
            compute_List=True,
            compute_Graph=True
        )

        # Save the super_index for this level
        super_index = torch.from_numpy(super_index.astype('int64'))
        d1.super_index = super_index

        # Create Data object for next level
        size = torch.LongTensor([c.shape[0] for c in cluster])
        pointer = torch.cat([torch.LongTensor([0]), size.cumsum(dim=0)])
        value = torch.cat([torch.from_numpy(x.astype('int64')) for x in cluster])
        pos = torch.from_numpy(x_c[:3].T)  # First 3 dimensions are position
        features_next = torch.from_numpy(x_c[3:].T)  # Rest are features
        
        # Create edges for next level
        s = torch.arange(edges[0].shape[0] - 1).repeat_interleave(
            torch.from_numpy((edges[0][1:] - edges[0][:-1]).astype("int64")))
        t = torch.from_numpy(edges[1].astype("int64"))
        edge_index_next = torch.vstack((s, t))
        edge_attr_next = torch.from_numpy(edges[2] / reg)

        # Compute node sizes for next level
        node_size_next = scatter_sum(
            d1.node_size if hasattr(d1, 'node_size') else torch.ones(d1.num_nodes, device=d1.pos.device),
            super_index,
            dim=0
        ).long()

        # Create next level Data object
        d2 = Data(
            pos=pos,
            rgb=features_next[:, :3],  # First 3 features are RGB
            linearity=features_next[:, 3:4],
            planarity=features_next[:, 4:5],
            scattering=features_next[:, 5:6],
            verticality=features_next[:, 6:7],
            elevation=features_next[:, 7:8],
            edge_index=edge_index_next,
            edge_attr=edge_attr_next,
            node_size=node_size_next
        )

        # Connect isolated nodes if any
        if d2.num_nodes > 1:
            d2 = d2.connect_isolated(k=k_adjacency)

        # Add to data list
        data_list.append(d2)

    # Use final level for predictions
    final_predictions = data_list[0].pos.new_zeros(data_list[0].num_nodes, dtype=torch.long)
    current_index = torch.arange(data_list[0].num_nodes, device=data_list[0].pos.device)
    
    # Traverse through hierarchy to get final predictions
    for d in data_list[:-1]:
        current_index = d.super_index[current_index]
    
    predictions = hard_assign(current_index, y)
    
    # Compute metrics
    accuracy = compute_mean_accuracy(predictions, y)
    miou = compute_miou(predictions, y, num_classes=13)
    duration = time.time() - start_time

    if verbose:
        print(f"Duration: {duration:.3f} seconds")
        print(f"Mean class accuracy: {accuracy:.3f}")
        print(f"Mean IoU: {miou:.3f}")

    # Update results dictionary with single-item lists for consistency
    results_dict["duration"] = duration
    results_dict["iterations"] = iterations
    results_dict["accuracy"] = accuracy
    results_dict["miou"] = miou
    results_dict["point_count"] = x.shape[0]
    results_dict["filename"] = str(filename)
    results_dict["regularization"] = regularization[-1]  # Last regularization value
    results_dict["spatial_weight"] = spatial_weight[-1]  # Last spatial weight value
    results_dict["cutoff"] = cutoff[-1]  # Last cutoff value
    results_dict["r_max"] = r_max
    results_dict["k_neighbors"] = k_neighbors
    results_dict["hierarchy_depth"] = len(regularization)
    results_dict["num_components"] = data_list[1].num_nodes

    return results_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test CutPursuit partitioning')
    parser.add_argument('--input', type=str, help='Input directory or path to LAS folder/file')
    parser.add_argument('--output', type=str, help='Output path for results CSV file')
    parser.add_argument('--reg_list', type=str, default="30", 
                       help='Comma-separated list of regularization parameters for hierarchy')
    parser.add_argument('--sw_list', type=str, default="1.0",
                       help='Comma-separated list of spatial weights for hierarchy')
    parser.add_argument('--cutoff_list', type=str, default="10",
                       help='Comma-separated list of cutoff values for hierarchy')
    parser.add_argument('--k_adjacency', type=int, default=5,
                       help='Number of neighbors to connect isolated nodes')
    parser.add_argument('--iterations', type=int, default=10, help='Maximum iterations')
    parser.add_argument('--parallel', default=True, action='store_true', help='Enable parallel processing')
    parser.add_argument('--verbose', default=True, action='store_true', help='Print debug information')
    parser.add_argument('--r_max', type=float, default=1.0, help='Maximum radius for neighbor search')
    parser.add_argument('--k_neighbors', type=int, default=50, help='Number of neighbors for geometric feature computation')
    args = parser.parse_args()

    # Process input path
    if Path(args.input).is_file():
        source_path = [args.input]
    elif Path(args.input).is_dir():
        source_path = [file for file in Path(args.input).iterdir() if file.is_file() and file.suffix == ".las"]
    else:
        raise ValueError(f"Invalid input path: {args.input}")

    # Set output path
    if args.output is None:
        output_path = Path(source_path[0]).parent / "cutpursuit_results.csv"
    else:
        output_path = args.output

    # Parse hierarchical parameters
    regularizations = [float(x) for x in args.reg_list.split(",")]
    spatial_weights = [float(x) for x in args.sw_list.split(",")]
    cutoffs = [int(x) for x in args.cutoff_list.split(",")]

    results_dfs = []

    # Process each file
    for file in source_path:
        print(f"Loading data from {file}")
        lasdata = laspy.read(file)
        x = torch.tensor(lasdata.xyz, dtype=torch.float32, device="cpu")
        y = torch.tensor(lasdata.classification.copy(), dtype=torch.int32, device="cpu")
        rgb = torch.tensor((np.concatenate([lasdata.red, lasdata.green, lasdata.blue], axis=0) / 65535).reshape(3, -1).T, dtype=torch.float32, device="cpu") / 255.0  # Normalize RGB to [0,1]
        
        print("\nInput data info:")
        print(f" - Number of points: {x.shape[0]}")
        print(f" - Dimensions: {x.shape[1]}")
        print(f" - Device: {x.device}")

        # Test different parameter combinations
        for reg in regularizations:
            for sw in spatial_weights:
                for cut in cutoffs:
                    print(f"\nTesting CutPursuit with reg={reg}, sw={sw}, cutoff={cut}, r_max={args.r_max}")
                    result_dict = test_cutpursuit(
                        filename=file,
                        x=x,
                        y=y,
                        rgb=rgb,
                        regularization=reg,
                        spatial_weight=sw,
                        cutoff=cut,
                        iterations=args.iterations,
                        r_max=args.r_max,
                        k_neighbors=args.k_neighbors,
                        k_adjacency=args.k_adjacency,
                        parallel=args.parallel,
                        verbose=args.verbose
                    )
                    results_dfs.append(pd.DataFrame(result_dict))

    # Create DataFrame and save to CSV
    results_df = pd.concat(results_dfs)
    
    # Reorder columns for better readability
    column_order = [
        'filename', 'point_count', 'accuracy', 'miou', 'duration', 'iterations',
        'num_components', 'regularization', 'spatial_weight', 'cutoff', 'r_max', 'k_neighbors', 'hierarchy_depth'
    ]
    results_df = results_df[column_order]
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\nResults Summary:")
    print(results_df.to_string())
    print(f"\nResults saved to: {output_path}") 