#include <torch/torch.h>
#include <iostream>
#include <random>
#include <chrono>
#include "gaussian_mixture.h"

// Generate synthetic data from a Gaussian Mixture
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
generate_gaussian_mixture_data(int n_samples, int n_components, int dim, int seed = 42) {
    torch::manual_seed(seed);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    
    // Generate random means
    auto means = torch::randn({n_components, dim}, options) * 4;
    
    // Generate random covariance matrices
    auto covs = torch::zeros({n_components, dim, dim}, options);
    for (int i = 0; i < n_components; i++) {
        auto A = torch::randn({dim, dim}, options);
        covs[i] = A.mm(A.t()) + torch::eye(dim, options) * 0.1;
    }
    
    // Generate samples
    auto samples_per_component = n_samples / n_components;
    auto data = torch::zeros({n_samples, dim}, options);
    auto labels = torch::zeros(n_samples, torch::TensorOptions().dtype(torch::kLong));
    
    for (int i = 0; i < n_components; i++) {
        auto start_idx = i * samples_per_component;
        auto end_idx = (i + 1) * samples_per_component;
        
        auto noise = torch::randn({samples_per_component, dim}, options);
        auto L = torch::linalg::cholesky(covs[i]);
        auto component_data = noise.mm(L.t()) + means[i].expand({samples_per_component, dim});
        
        data.slice(0, start_idx, end_idx) = component_data;
        labels.slice(0, start_idx, end_idx).fill_(i);
    }
    
    return std::make_tuple(data, labels, means, covs);
}

void print_tensor_info(const torch::Tensor& tensor, const std::string& name) {
    std::cout << name << ":\n";
    std::cout << " - Shape: [";
    for (size_t i = 0; i < tensor.dim(); ++i) {
        std::cout << tensor.size(i);
        if (i < tensor.dim() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << " - Device: " << tensor.device() << "\n";
    std::cout << " - Type: " << tensor.dtype() << "\n";
    std::cout << " - Is contiguous: " << tensor.is_contiguous() << "\n\n";
}

bool verify_shapes(const torch::Tensor& data, 
                  const torch::Tensor& pi, 
                  const torch::Tensor& mu, 
                  const torch::Tensor& sigma, 
                  const torch::Tensor& labels,
                  int n_samples,
                  int n_components,
                  int dim) {
    bool shapes_ok = true;
    
    // Check data shape
    if (data.sizes() != torch::IntArrayRef({n_samples, dim})) {
        std::cout << "ERROR: Incorrect data shape. Expected [" << n_samples << ", " << dim 
                  << "], got " << data.sizes() << "\n";
        shapes_ok = false;
    }
    
    // Check pi shape
    if (pi.sizes() != torch::IntArrayRef({n_components})) {
        std::cout << "ERROR: Incorrect pi shape. Expected [" << n_components 
                  << "], got " << pi.sizes() << "\n";
        shapes_ok = false;
    }
    
    // Check mu shape
    if (mu.sizes() != torch::IntArrayRef({n_components, dim})) {
        std::cout << "ERROR: Incorrect mu shape. Expected [" << n_components << ", " << dim 
                  << "], got " << mu.sizes() << "\n";
        shapes_ok = false;
    }
    
    // Check sigma shape
    if (sigma.sizes() != torch::IntArrayRef({n_components, dim, dim})) {
        std::cout << "ERROR: Incorrect sigma shape. Expected [" << n_components << ", " << dim << ", " << dim 
                  << "], got " << sigma.sizes() << "\n";
        shapes_ok = false;
    }
    
    // Check labels shape
    if (labels.sizes() != torch::IntArrayRef({n_samples})) {
        std::cout << "ERROR: Incorrect labels shape. Expected [" << n_samples 
                  << "], got " << labels.sizes() << "\n";
        shapes_ok = false;
    }
    
    return shapes_ok;
}

void run_gmm_test(int n_samples, int n_components, int dim, int subsample_size = -1) {
    std::cout << "\n=== Testing with " << n_samples << " samples, " 
              << n_components << " components, " 
              << dim << " dimensions";
    if (subsample_size > 0) {
        std::cout << ", subsample size " << subsample_size;
    } else {
        std::cout << ", no subsampling";
    }
    std::cout << " ===\n";
    
    std::cout << "Generating synthetic data...\n";
    auto [data, true_labels, true_means, true_covs] = generate_gaussian_mixture_data(
        n_samples, n_components, dim);
    
    // Print input tensor information
    print_tensor_info(data, "Input data");

    auto start_time = std::chrono::high_resolution_clock::now();
      
    // Test GPU if available
    if (torch::cuda::is_available()) {
        std::cout << "\nTesting on GPU...\n";
        auto data_gpu = data.cuda();
        start_time = std::chrono::high_resolution_clock::now();
        
        try {
            auto [pi_gpu, mu_gpu, sigma_gpu, labels_gpu] = gmm_core_implementation(
                data_gpu,
                n_components,
                1.0f,
                1e-2f,
                10,
                subsample_size
            );
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);
            
            std::cout << "GPU Time: " << duration.count() / 1000.0 << " seconds\n";
            
            // Verify shapes
            bool shapes_ok = verify_shapes(data_gpu, pi_gpu, mu_gpu, sigma_gpu, labels_gpu,
                                         n_samples, n_components, dim);
            std::cout << "Shape verification: " << (shapes_ok ? "PASSED" : "FAILED") << "\n";
            
        } catch (const std::exception& e) {
            std::cout << "GPU Error: " << e.what() << std::endl;
            return;
        }
    }
}

int main() {
    // Test cases with different sizes and dimensions
    // Format: {n_samples, n_components, dim, subsample_size}
    std::vector<std::tuple<int, int, int, int>> test_cases = {
        {100000, 1000, 3, -1},
    };
    
    for (const auto& [n_samples, n_components, dim, subsample_size] : test_cases) {
        run_gmm_test(n_samples, n_components, dim, subsample_size);
    }
    
    return 0;
}