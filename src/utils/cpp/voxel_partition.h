#include <torch/torch.h>

namespace voxel_partition {
    torch::Tensor voxelize_points_mt(
        const torch::Tensor& points,
        const std::vector<double>& voxelSizes
    );
    torch::Tensor assign_parents(
        const torch::Tensor& children_labels,
        torch::Tensor parent_labels
    );
    torch::Tensor voxelize_points_with_size(
        const torch::Tensor& points,
        double voxelSize
    );
}
