#include "voxel_partition.h"
#include <torch/extension.h>
#include <ATen/Parallel.h>

namespace voxel_partition {

/**
 * @brief Voxelise un nuage de points 3D en labels contigus, multi-thread.
 *
 * @param points    Tensor CPU de forme (N, 3), dtype float32
 * @param voxelSize Taille du voxel (scalaire double)
 * @return Tensor int64 de forme (N,), labels contigus [0..K-1]
 */
torch::Tensor voxelize_points_with_size(
    const torch::Tensor& points,
    double voxelSize
) {
    TORCH_CHECK(points.device().is_cpu(), "points must be a CPU tensor");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
                "points must have shape (N,3)");

    // 1) Discrétisation des coordonnées
    // points / voxelSize => arrondi vers le bas
    float voxelSizeFloat = static_cast<float>(voxelSize); // Convert double to float for internal use
    torch::Tensor discrete = (points / voxelSizeFloat).floor().to(torch::kInt64);

    // 2) Calcul des bornes min et max
    // dim 0 = rows: N, dim 1 = xyz
    auto min_max = at::aminmax(discrete, /*dim=*/0, /*keepdim=*/false);
    torch::Tensor coords_min = std::get<0>(min_max);
    torch::Tensor coords_max = std::get<1>(min_max);

    // 3) Ajustement pour rendre coordonnées positives
    torch::Tensor coords_shifted = discrete - coords_min;
    torch::Tensor dims = (coords_max - coords_min + 1).to(torch::kInt64);

    // 4) Encodage 3D -> 1D (clé) en multi-thread
    int64_t N = points.size(0);
    torch::Tensor keys = torch::empty({N}, torch::kInt64);
    int64_t dim0 = dims[0].item<int64_t>();
    int64_t dim1 = dims[1].item<int64_t>();

    torch::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
        auto src = coords_shifted.accessor<int64_t,2>();
        auto dst = keys.accessor<int64_t,1>();
        for (int64_t i = start; i < end; ++i) {
            int64_t x = src[i][0];
            int64_t y = src[i][1];
            int64_t z = src[i][2];
            dst[i] = x + y * dim0 + z * dim0 * dim1;
        }
    });

    // 5) Unique + inverse pour labels contigus, trié
    // return_inverse = true => tuple (unique, inverse)
    auto unique_res = torch::_unique(keys, /*sorted=*/true,
                                 /*return_inverse=*/true);
    torch::Tensor labels = std::get<1>(unique_res);

    return labels;
}

torch::Tensor assign_parents(
    const torch::Tensor& children_labels,
    torch::Tensor parent_labels
) {
    // Get unique values in labels
    auto unique_labels = std::get<0>(torch::_unique(children_labels, /*sorted=*/true));
    
    for (int64_t i = 0; i < unique_labels.size(0); i++) {
        auto unique_label = unique_labels[i];
        auto parents = parent_labels[children_labels == unique_label];
        if (torch::numel(std::get<0>(torch::_unique(parents))) > 1) {
            // Get the most represented parent
            auto most_rep = std::get<0>(torch::_unique(parents))[torch::argmax(std::get<1>(torch::_unique(parents, /*return_counts=*/true)))];
            parent_labels[children_labels == unique_label] = most_rep;
        }
    }
    return parent_labels;
}
    
    

torch::Tensor voxelize_points_mt(
    const torch::Tensor& points,
    const std::vector<double>& voxelSizes
) {
    int64_t num_points = points.size(0);
    int64_t num_levels = static_cast<int64_t>(voxelSizes.size());
    
    // Create output tensor of shape [num_points, num_levels]
    auto labels = torch::zeros({num_points, num_levels}, torch::kInt64);
    
    // Process each level
    for (size_t i = 0; i < voxelSizes.size(); i++) {
        // Get voxel size for this level
        double voxelSize = voxelSizes[i];
        
        // Voxelize the points for this level
        auto level_labels = voxelize_points_with_size(points, voxelSize);
        
        // Copy level labels to the corresponding column in output tensor
        for (int64_t j = 0; j < num_points; j++) {
            labels[j][i] = level_labels[j];
        }
        
        // For levels > 0, assign parent labels
        if (i > 0) {
            // Create copies of columns to work with
            torch::Tensor current_level = labels.slice(1, i, i+1).reshape({num_points});
            torch::Tensor prev_level = labels.slice(1, i-1, i).reshape({num_points});
            
            // Get updated labels with parent assignments
            torch::Tensor updated_labels = assign_parents(current_level, prev_level.clone());
            
            // Copy updated labels back to output tensor
            for (int64_t j = 0; j < num_points; j++) {
                labels[j][i] = updated_labels[j];
            }
        }
    }
    
    return labels;
}



} // namespace voxel_partition

// Register the function with PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize_points_mt", &voxel_partition::voxelize_points_mt, 
          "Voxelize a point cloud into contiguous labels");
    m.def("voxelize_points_with_size", &voxel_partition::voxelize_points_with_size,
          "Voxelize a point cloud with a specific voxel size");
}