from voxel_partition_cpp import voxelize_points_with_size
import torch
from typing import List


__all__ = ['voxelize_points']
def voxelize_points(points: torch.Tensor, voxel_size: List[float]) -> torch.Tensor:
    labels = torch.zeros((points.shape[0], len(voxel_size)), dtype=torch.int64)
    for i, vs in enumerate(voxel_size):
        # Process this level
        level_labels = voxelize_points_with_size(points, float(vs))
        labels[:, i] = level_labels

        # if i > 0:
        #     # For each unique label in current level
        #     for label in torch.unique(level_labels):
        #         mask = level_labels == label
        #         # Get parents for these points
        #         parents = labels[mask, i-1]
        #         # If multiple parents, pick the most frequent one
        #         if len(torch.unique(parents)) > 1:
        #             unique_parents, counts = torch.unique(parents, return_counts=True)
        #             most_common_parent = unique_parents[torch.argmax(counts)]
        #             # Assign this parent to all points with this label
        #             labels[mask, i - 1] = most_common_parent
    return labels
