#ifndef QEM_PARTITIONING_H
#define QEM_PARTITIONING_H

#include <torch/torch.h>
#include <queue>
#include <unordered_map>
#include <vector>

namespace priority_queue_ns {
    struct MinHeapElement {
        int index;
        int candidacy;
        double cost;
        
        bool operator<(const MinHeapElement& other) const {
            return cost > other.cost; // Min-heap
        }
    };
    
    struct MaxHeapElement {
        int cluster_index;  // Index of the cluster
        int point_index;    // Index of the potential generator point
        double error;       // Quadric error
        
        bool operator<(const MaxHeapElement& other) const {
            return error < other.error; // Max-heap
        }
    };
    
    class MinQueue {
    private:
        std::priority_queue<MinHeapElement> min_queue;
        std::unordered_map<int, double> cost_map;
    public:
        void update(int index, int candidacy, double cost);
        void batch_update(const std::vector<std::tuple<int, int, double>>& updates);
        void batch_update(const std::vector<std::tuple<int64_t, int, double>>& updates);
        void exclude(int index);
        std::tuple<int, int, double> pop();
        std::vector<std::tuple<int, int, double>> pop_batch(int max_count);
        bool empty();
        size_t size() const;
        void print();
        void clear();
    };

    class MaxQueue {
    private:
        std::priority_queue<MaxHeapElement> max_queue;
        std::unordered_map<int, double> error_map;
    public:
        void add(int cluster_index, int point_index, double error);
        std::tuple<int, int, double> pop();
        bool empty();
        size_t size() const;
        void clear();
    };
}

namespace qem {
    torch::Tensor compute_quadric(const torch::Tensor& point, const torch::Tensor& normals);
    float compute_area(const torch::Tensor& distances);
    torch::Tensor compute_diffused_quadrics(const int64_t idx_i, const torch::Tensor& points, const torch::Tensor& neighbors, const torch::Tensor& distances, const torch::Tensor& normals);
    float compute_cost(const torch::Tensor& point, const torch::Tensor& generator, const torch::Tensor& quadric, const float& reg);
    void region_growing(const torch::Tensor& points, const torch::Tensor& neighbors, const torch::Tensor& distances, const torch::Tensor& quadrics, const torch::Tensor& generators, torch::Tensor& assignments, const float& reg);
    
    // Helper functions for cluster analysis
    torch::Tensor compute_optimal_generator(const torch::Tensor& Q);
    int64_t find_closest_point(const torch::Tensor& points, const torch::Tensor& target_point);
    double compute_quadric_error(const torch::Tensor& point, const torch::Tensor& Q);
    std::tuple<int64_t, double> find_max_error_point(const torch::Tensor& points, const torch::Tensor& Q);
    priority_queue_ns::MaxQueue analyze_and_update_clusters(const torch::Tensor& points, torch::Tensor& generators, const torch::Tensor& assignments, const torch::Tensor& quadrics, const float& qem_tol);
    
    // Main function
    std::tuple<torch::Tensor, torch::Tensor> qem_partitioning(const torch::Tensor& points, const torch::Tensor& neighbors, const torch::Tensor& distances, const torch::Tensor& normals, const int& k_init, const float& reg, const float& qem_tol, const int max_iterations = 5);
}

#endif // QEM_PARTITIONING_H