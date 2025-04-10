#include <torch/torch.h>
#include <cmath>
#include <vector>
#include <tuple>
#include <queue>
#include <unordered_map>
#include <limits>
#include <omp.h>
#include <mutex>

// Define a new namespace for the priority queue
namespace priority_queue_ns {

    struct MinHeapElement {
        int index;
        int candidacy;
        double cost;

        bool operator<(const MinHeapElement& other) const {
            return cost > other.cost;
        }
    };
    
    struct MaxHeapElement {
        int cluster_index;
        int point_index;
        double error;
        
        bool operator<(const MaxHeapElement& other) const {
            return error < other.error; // Max-heap
        }
    };

    class MinQueue {
    private:
        std::priority_queue<MinHeapElement> min_queue;
        std::unordered_map<int, double> cost_map;

    public:
        void update(int index, int candidacy, double cost) {
            auto it = cost_map.find(index);
            if (it != cost_map.end() && it->second <= cost) {
                // Only update if new cost is better (lower)
                return;
            }
            
            // Store new cost and add to queue
            cost_map[index] = cost;
            MinHeapElement element{index, candidacy, cost};
            min_queue.push(element);
        }

        // Batch update to reduce overhead
        void batch_update(const std::vector<std::tuple<int, int, double>>& updates) {
            for (const auto& [idx, candidacy, cost] : updates) {
                auto it = cost_map.find(idx);
                if (it == cost_map.end() || it->second > cost) {
                    cost_map[idx] = cost;
                    MinHeapElement element{idx, candidacy, cost};
                    min_queue.push(element);
                }
            }
        }
        
        // Overload for int64_t indices
        void batch_update(const std::vector<std::tuple<int64_t, int, double>>& updates) {
            for (const auto& [idx, candidacy, cost] : updates) {
                auto it = cost_map.find(static_cast<int>(idx));
                if (it == cost_map.end() || it->second > cost) {
                    cost_map[static_cast<int>(idx)] = cost;
                    MinHeapElement element{static_cast<int>(idx), candidacy, cost};
                    min_queue.push(element);
                }
            }
        }

        void exclude(int index) {
            cost_map.erase(index);
            // Lazy removal - will filter when popping
        }

        std::tuple<int, int, double> pop() {
            while (!min_queue.empty()) {
                MinHeapElement top = min_queue.top();
                min_queue.pop();
                
                auto it = cost_map.find(top.index);
                // Check if this element is still valid and has the current best cost
                if (it != cost_map.end() && std::abs(it->second - top.cost) < 1e-10) {
                    cost_map.erase(it); // Remove from map
                    return std::make_tuple(top.index, top.candidacy, top.cost);
                }
                // Otherwise this is an outdated entry, skip it
            }
            return std::make_tuple(-1, -1, std::numeric_limits<double>::infinity());
        }

        // Pop multiple elements at once to amortize overhead
        std::vector<std::tuple<int, int, double>> pop_batch(int max_count) {
            std::vector<std::tuple<int, int, double>> result;
            result.reserve(max_count);
            
            for (int i = 0; i < max_count && !cost_map.empty(); i++) {
                auto [idx, candidacy, cost] = pop();
                if (idx < 0) break;
                result.emplace_back(idx, candidacy, cost);
            }
            
            return result;
        }

        bool empty() {
            return cost_map.empty();
        }

        size_t size() const {
            return cost_map.size();
        }

        void print() {
            auto temp_queue = min_queue;
            while (!temp_queue.empty()) {
                MinHeapElement top = temp_queue.top();
                std::cout << "Index: " << top.index << ", Candidacy: " << top.candidacy 
                         << ", Cost: " << top.cost << std::endl;
                temp_queue.pop();
            }
        }
        
        void clear() {
            while (!min_queue.empty()) min_queue.pop();
            cost_map.clear();
        }
    };

    class MaxQueue {
    private:
        std::priority_queue<MaxHeapElement> max_queue;
        std::unordered_map<int, double> error_map;

    public:
        void add(int cluster_index, int point_index, double error) {
            auto it = error_map.find(cluster_index);
            if (it != error_map.end() && it->second >= error) {
                // Only update if new error is worse (higher)
                return;
            }
            
            // Store new error and add to max queue
            error_map[cluster_index] = error;
            MaxHeapElement element{cluster_index, point_index, error};
            max_queue.push(element);
        }
        
        std::tuple<int, int, double> pop() {
            if (max_queue.empty()) {
                return std::make_tuple(-1, -1, 0.0);
            }
            
            MaxHeapElement top = max_queue.top();
            max_queue.pop();
            error_map.erase(top.cluster_index);
            return std::make_tuple(top.cluster_index, top.point_index, top.error);
        }

        bool empty() {
            return error_map.empty();
        }
        
        size_t size() const {
            return error_map.size();
        }
        
        void clear() {
            while (!max_queue.empty()) max_queue.pop();
            error_map.clear();
        }
    };
}

namespace qem {

    torch::Tensor compute_quadric(const torch::Tensor& point, const torch::Tensor& normal) {
        // The plane equation is ax + by + cz + d = 0
        // where (a,b,c) is the normal vector and d = -(ax₀ + by₀ + cz₀)
        torch::Tensor d = -normal.dot(point);
        
        // Create plane equation vector [a, b, c, d]
        torch::Tensor plane = torch::cat({normal, d.unsqueeze(0)}, 0);
        
        // Create quadric matrix properly using outer product
        return plane.unsqueeze(1) * plane.unsqueeze(0);
    }

    float compute_area(const torch::Tensor& distances) {
        float sum_dist = distances.sum().item<float>();
        return (sum_dist * sum_dist) / (2 * distances.size(0) * distances.size(0));
    }

    // Cache for storing precomputed quadrics to avoid redundant calculations
    class QuadricCache {
    private:
        std::unordered_map<int64_t, torch::Tensor> cache;
        std::mutex mutex;
        
    public:
        torch::Tensor get_or_compute(int64_t idx, const torch::Tensor& points, const torch::Tensor& normals) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto it = cache.find(idx);
                if (it != cache.end()) {
                    return it->second;
                }
            }
            
            // Compute if not found
            torch::Tensor result = compute_quadric(points[idx], normals[idx]);
            
            // Store in cache
            {
                std::lock_guard<std::mutex> lock(mutex);
                cache[idx] = result;
            }
            
            return result;
        }
        
        void clear() {
            std::lock_guard<std::mutex> lock(mutex);
            cache.clear();
        }
    };
    
    // Global quadric cache to reduce redundant calculations
    static QuadricCache quadric_cache;
    
    // Optimized diffused quadrics computation with caching and better memory usage
    torch::Tensor compute_diffused_quadrics(
        const int64_t idx_i, 
        const torch::Tensor& points, 
        const torch::Tensor& neighbors, 
        const torch::Tensor& distances, 
        const torch::Tensor& normals) {
        
        int64_t k = neighbors.size(1);
        torch::Tensor idx_neighbors = neighbors[idx_i]; // neighbors of i
        
        // Pre-allocate result tensor
        torch::Tensor diffused_quadrics = torch::zeros({4, 4}, torch::kFloat32);
        
        // Pre-compute and cache intermediate results
        std::vector<float> support_areas(k);
        std::vector<int64_t> neighbor_indices(k);
        
        // First pass: collect indices and compute areas
        for (int64_t j = 0; j < k; j++) {
            neighbor_indices[j] = idx_neighbors[j].item<int64_t>();
            torch::Tensor j_neighbors = neighbors[neighbor_indices[j]];
            support_areas[j] = compute_area(distances.index_select(0, j_neighbors));
        }
        
        // Second pass: accumulate quadrics (better cache locality)
        for (int64_t j = 0; j < k; j++) {
            int64_t j_idx = neighbor_indices[j];
            // Get quadric from cache or compute if not available
            torch::Tensor j_quadric = quadric_cache.get_or_compute(j_idx, points, normals);
            
            // Multiply and accumulate directly (avoid creating another temporary tensor)
            diffused_quadrics += support_areas[j] * j_quadric;
        }
        
        return diffused_quadrics;
    }


    // Optimized cost computation function to reduce tensor operations overhead
    float compute_cost(const torch::Tensor& point, const torch::Tensor& generator, const torch::Tensor& quadric, const float& reg) {
        // Pre-allocate the homogeneous vector to avoid repeated allocations
        static thread_local torch::Tensor vect = torch::zeros({4}, torch::kFloat32);
        
        // Copy generator values directly (faster than cat operation)
        vect[0] = generator[0].item<float>();
        vect[1] = generator[1].item<float>();
        vect[2] = generator[2].item<float>();
        vect[3] = 1.0f;
        
        // Compute quadric term efficiently
        float quadric_term = (vect.dot(quadric.mv(vect))).item<float>();
        
        // Compute Euclidean term manually to avoid temporary tensor creation
        float dx = point[0].item<float>() - generator[0].item<float>();
        float dy = point[1].item<float>() - generator[1].item<float>();
        float dz = point[2].item<float>() - generator[2].item<float>();
        float euclidean_term = reg * (dx*dx + dy*dy + dz*dz);
        return euclidean_term + quadric_term;
    }

    void region_growing(
        const torch::Tensor& points,
        const torch::Tensor& neighbors,
        const torch::Tensor& distances,
        const torch::Tensor& quadrics,
        const torch::Tensor& generators,
        torch::Tensor& assignments,
        const float& reg) {

        // Single mutex for the critical section
        std::mutex pq_mutex;

        // Priority queue for region growing
        priority_queue_ns::MinQueue pq;
        
        // Pre-allocate vector for unassigned indices
        std::vector<int64_t> unassigned_indices;
        unassigned_indices.reserve(neighbors.size(1));

        int iteration = 0;
        auto loop_start = std::chrono::high_resolution_clock::now();
        
        // Ensure each generator is assigned to its own cluster and pushed to min queue
        #pragma omp parallel for
        for (int64_t i = 0; i < generators.size(0); i++) {
            int64_t gen_idx = generators[i].item<int64_t>();
            assignments[gen_idx] = i;
            pq.update(gen_idx, i, 0.0);
        }

        while (!pq.empty()) {
            // Simple single item pop
            auto [top_index, top_candidacy, top_cost] = pq.pop();
            
            if (assignments[top_index].item<int64_t>() == -1) {
                assignments[top_index] = top_candidacy;
            }
            
            torch::Tensor top_neighbors = neighbors[top_index];
            
            // Get generator point directly
            int64_t gen_idx = generators[top_candidacy].item<int64_t>();
            torch::Tensor gen_point = points[gen_idx];
            
            // Find unassigned neighbors
            unassigned_indices.clear();
            for (int64_t j = 0; j < top_neighbors.size(0); j++) {
                int64_t j_idx = top_neighbors[j].item<int64_t>();
                if (assignments[j_idx].item<int64_t>() == -1) {
                    unassigned_indices.push_back(j_idx);
                }
            }
            
            if (unassigned_indices.empty()) continue;
            
            #pragma omp parallel for
            for (size_t i = 0; i < unassigned_indices.size(); i++) {
                int64_t j_idx = unassigned_indices[i];
                float cost = compute_cost(points[j_idx], gen_point, quadrics[j_idx], reg);
                
                // Use a mutex to protect the queue update
                std::lock_guard<std::mutex> lock(pq_mutex);
                pq.update(j_idx, top_candidacy, cost);
            }
            iteration++;
            if (iteration % 10000 == 0) {
                auto current = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current - loop_start);
                std::cout << "Processed " << iteration << " points in " << duration.count() / 1000.0 << " seconds";
                std::cout << " (Queue size: " << pq.size() << ")" << std::endl;
                loop_start = std::chrono::high_resolution_clock::now();
            }
        }
    }
    
    // Compute the optimal generator for a cluster using matrix inversion or SVD
    torch::Tensor compute_optimal_generator(const torch::Tensor& Q) {
        // Extract 3x3 submatrix and vector from Q
        torch::Tensor A = Q.slice(0, 0, 3).slice(1, 0, 3);
        torch::Tensor b = Q.slice(0, 0, 3).slice(1, 3, 4).squeeze(1);
        
        // Try to solve the linear system A * x = -b
        torch::Tensor opt_gen;
        try {
            opt_gen = -torch::linalg::solve(A, b, false);
        } catch (const std::exception& e) {
            // If not invertible, use SVD solver
            auto [U, S, V] = torch::linalg::svd(A, /*full_matrices=*/true, std::nullopt);
            torch::Tensor Sinv = torch::zeros_like(S);
            for (int64_t j = 0; j < S.size(0); j++) {
                if (S[j].item<float>() > 1e-10) {
                    Sinv[j] = 1.0 / S[j];
                }
            }
            torch::Tensor Ainv = V.transpose(-2, -1) * 
                                 Sinv.unsqueeze(-1) * 
                                 U.transpose(-2, -1);
            opt_gen = -torch::matmul(Ainv, b);
        }
        return opt_gen;
    }

    int64_t find_closest_point(const torch::Tensor& points, 
                              const torch::Tensor& target_point) {
        torch::Tensor distances = torch::norm(points - target_point.unsqueeze(0), 2, 1);
        torch::Tensor min_idx = torch::argmin(distances, 0);
        return min_idx.item<int64_t>();
    }

    std::tuple<int64_t, double> find_max_error_point(const torch::Tensor& points, 
                                                    const torch::Tensor& Q) {

        
        // Compute quadric error for each point
        torch::Tensor homog_points = torch::cat({points, 
            torch::ones({points.size(0), 1}, points.options())}, 1);
        torch::Tensor errors = torch::matmul(torch::matmul(homog_points, Q), homog_points.transpose(0, 1))
            .diagonal();
        
        // Find max error and corresponding index
        torch::Tensor max_error_idx = torch::argmax(errors, 0);
        double max_error = errors[max_error_idx].item<double>();
    
        int64_t local_idx = max_error_idx.item<int64_t>();
        
        return std::make_tuple(local_idx, max_error);
    }

    // Analyze clusters, update generators, find potential generators and add them to max queue
    priority_queue_ns::MaxQueue analyze_and_update_clusters(
        const torch::Tensor& points,
        torch::Tensor& generators,
        const torch::Tensor& assignments,
        const torch::Tensor& quadrics,
        const float& qem_tol) {

        // Max queue for potential splits
        priority_queue_ns::MaxQueue max_queue;

        auto loop_start = std::chrono::high_resolution_clock::now();
        
        // Tracks if all clusters satisfy tolerance
        bool all_satisfied = true;
        
        #pragma omp parallel for reduction(max:all_satisfied)
        for (int64_t i = 0; i < generators.size(0); i++) {
            auto mask_i = assignments == i;
            torch::Tensor global_indices = torch::nonzero(mask_i).squeeze(1);

            if (global_indices.size(0) == 0) {
                std::cout << "No points in cluster " << i << std::endl;
                continue;
            }

            torch::Tensor Q = quadrics[generators[i].item<int64_t>()];
            torch::Tensor opt_gen_i = compute_optimal_generator(Q);
            // Find closest point to optimal generator
            int64_t closest_idx = global_indices[find_closest_point(points.index({global_indices}), opt_gen_i)].item<int64_t>();

            // Update generator
            generators[i] = closest_idx;
            
            auto [local_idx, error] = find_max_error_point(points.index({global_indices}), Q);
            int64_t max_error_idx = global_indices[local_idx].item<int64_t>();
            
            if (error > qem_tol) {
                all_satisfied = false;
                
                #pragma omp critical
                {
                    // Candidate generator to split
                    max_queue.add(i, max_error_idx, error);
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - loop_start);
        std::cout << "Cluster analysis took " << duration.count() / 1000.0 << " seconds" << std::endl;
        return max_queue;
    }

    std::tuple<torch::Tensor, torch::Tensor> qem_partitioning(
        const torch::Tensor& points, 
        const torch::Tensor& neighbors, 
        const torch::Tensor& distances, 
        const torch::Tensor& normals, 
        const int& k_init, 
        const float& reg, 
        const float& qem_tol,
        const int max_iterations) {
        
        int64_t N = points.size(0);
        torch::Tensor generators = torch::randperm(N).slice(0, 0, k_init);
        torch::Tensor assignments = -torch::ones({N}, torch::kInt64);
        torch::Tensor quadrics = torch::zeros({N, 4, 4}, torch::kFloat32);
        
        int iteration = 0;
        bool all_satisfied = false;
        
        // Compute all quadrics once at the beginning - they won't change during iterations
        auto start_quadrics = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (int64_t i = 0; i < N; i++) {
            quadrics[i] = compute_diffused_quadrics(i, points, neighbors, distances, normals);
        }

        auto end_quadrics = std::chrono::high_resolution_clock::now();
        auto duration_quadrics = std::chrono::duration_cast<std::chrono::milliseconds>(end_quadrics - start_quadrics);

        std::cout << "Initialization took " << duration_quadrics.count() / 1000.0 << " seconds" << std::endl;
        
        if (max_iterations == 0) {
            region_growing(points, neighbors, distances, quadrics, generators, assignments, reg);
            return std::make_tuple(generators, assignments);
        }

        while (iteration < max_iterations && !all_satisfied) {
            std::cout << "Iteration " << iteration << ", current clusters: " << generators.size(0) << std::endl;

            assignments.fill_(-1);

            // Execute region growing algorithm
            region_growing(points, neighbors, distances, quadrics, generators, assignments, reg);

            // Analyze clusters, update generators, and add potential generators to max queue
            auto max_queue = analyze_and_update_clusters(
                points, generators, assignments, quadrics, qem_tol);
            
            // Check if any clusters need to be split
            all_satisfied = max_queue.empty();
            
            // TODO: Self-contained function for batch splitting policy
            auto start = std::chrono::high_resolution_clock::now();
            
            // Keep track of clusters to split and their neighbors
            std::vector<int> clusters_to_split;
            std::unordered_set<int> excluded_clusters;
            std::vector<int64_t> potential_gens_to_add;
            
            // Select clusters to split greedily
            while (!max_queue.empty()) {

                auto [cluster_idx, point_idx, error] = max_queue.pop();
                if (excluded_clusters.count(cluster_idx) > 0) {
                    continue;
                }
                
                clusters_to_split.push_back(cluster_idx);
                potential_gens_to_add.push_back(point_idx);
                excluded_clusters.insert(cluster_idx);
                
                // Find neighboring clusters and exclude them
                for (int64_t i = 0; i < N; i++) {
                    if (assignments[i].item<int64_t>() == cluster_idx) {
                        torch::Tensor idx_neighbors = neighbors[i];
                        for (int64_t j = 0; j < idx_neighbors.size(0); j++) {
                            int64_t neigh_idx = idx_neighbors[j].item<int64_t>();
                            int64_t neigh_cluster = assignments[neigh_idx].item<int64_t>();
                            if (neigh_cluster >= 0 && neigh_cluster != cluster_idx) {
                                excluded_clusters.insert(neigh_cluster);
                            }
                        }
                    }
                }
            }
            
            // If no clusters to split, jump to next iteration
            if (clusters_to_split.empty()) {
                std::cout << "No clusters to split" << std::endl;
            }
            else {
                std::cout << "Selected " << clusters_to_split.size() << " clusters to split" << std::endl;
                torch::Tensor new_generators = torch::zeros({generators.size(0) + clusters_to_split.size()}, torch::kInt64);
                new_generators.slice(0, 0, generators.size(0)).copy_(generators);
                
                for (size_t i = 0; i < potential_gens_to_add.size(); i++) {
                    new_generators[generators.size(0) + i] = potential_gens_to_add[i];
                }
                
                generators = new_generators;
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cout << "Batch splitting took " << duration.count() / 1000.0 << " seconds" << std::endl;
                
            }
            iteration++;
        }
        // Final region growing for final assignment
        assignments.fill_(-1);
        region_growing(points, neighbors, distances, quadrics, generators, assignments, reg);
        return std::make_tuple(generators, assignments);
    }
}
