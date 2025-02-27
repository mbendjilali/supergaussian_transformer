#pragma once
#include <torch/torch.h>
#include <vector>

namespace gmm {
    enum class GMMVariant {
        GEM,    // Generalized EM
        VBEM,   // Variational Bayes EM
        CEM     // Classification EM
    };

    // Core implementations for different variants
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
    gem_core(
        torch::Tensor x,
        int K,
        float alpha,
        float tol,
        int max_iter
    );

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
    vbem_core(
        torch::Tensor x,
        int K,
        float alpha,
        float tol,
        int max_iter
    );

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
    cem_core(
        torch::Tensor x,
        int K,
        float alpha,
        float tol,
        int max_iter
    );

    // Hierarchical GMM declaration with variant selection
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> 
    gmm_hierarchical_implementation(
        torch::Tensor x,
        torch::Tensor hierarchy_k,
        float alpha,
        float tol,
        int max_iter,
        GMMVariant variant = GMMVariant::GEM
    );
}