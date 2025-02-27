#include "gaussian_mixture.h"
#include <cmath>
#include <iostream>
#include <torch/torch.h>

namespace gmm {
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
    gem_core(
        torch::Tensor x,
        int K,
        float alpha,
        float tol,
        int max_iter
    ) {
        torch::NoGradGuard no_grad;
        try {
            const int N = x.size(0);
            const int D = x.size(1);
            const auto device = x.device();
            const auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
            const auto idx_options = torch::TensorOptions().device(device).dtype(torch::kInt64);

            // Increase regularization constant
            const float reg_const = 1e-3;  // Increased from 1e-4

            // Adjust K if necessary
            K = std::min(K, N);

            // Initialize model parameters
            auto pi = torch::full({K}, alpha, options).div(K * alpha);  // Uniform initial weights
            
            // Initialize means by randomly selecting K points from the data
            auto indices = torch::randperm(N, idx_options).slice(0, 0, K);
            auto mu = x.index_select(0, indices);  // Select K random points as initial centroids
            
            auto sigma = torch::eye(D, options).repeat({K, 1, 1});     // Identity initial covariances
            
            // Pre-allocate tensors for EM algorithm
            auto resp = torch::empty({N, K}, options);
            auto log_resp = torch::empty({N, K}, options);
            auto L_batch = torch::empty({K, D, D}, options);
            auto log_det = torch::empty({K}, options);
            
            for(int iter = 0; iter < max_iter; iter++) {
                // Before Cholesky decomposition, ensure matrix is positive definite
                auto sigma_reg = sigma.clone();
                for(int k = 0; k < K; k++) {
                    // Add larger regularization to diagonal
                    sigma_reg[k].add_(torch::eye(D, options).mul(reg_const));
                    
                    // Ensure symmetry
                    sigma_reg[k] = (sigma_reg[k] + sigma_reg[k].transpose(-2, -1)) / 2.0;
                    
                    // Add minimum eigenvalue regularization if needed
                    auto eigh_result = torch::linalg::eigh(sigma_reg[k], "L");  // "L" for lower triangular
                    auto eigenvalues = std::get<0>(eigh_result);
                    auto min_eig = eigenvalues[0].item<float>();
                    if (min_eig < reg_const) {
                        sigma_reg[k].add_(torch::eye(D, options).mul(reg_const - min_eig + 1e-6));
                    }
                }
                
                // Use regularized covariance for Cholesky
                L_batch = torch::linalg::cholesky((sigma_reg + sigma_reg.transpose(-2, -1))/2);
                log_det = 2 * L_batch.diagonal(0, -2, -1).log().sum(-1);
                auto diff = x.unsqueeze(1) - mu.unsqueeze(0);  // [N, K, D]
                
                for(int k = 0; k < K; k++) {
                    // Compute log-likelihood for component k
                    auto solved = at::linalg_solve_triangular(
                        L_batch[k], diff.select(1, k).t(),
                        /*upper=*/false, /*transpose=*/true, /*unitriangular=*/false
                    );
                    auto sq_mahalanobis = solved.t().pow(2).sum(1);
                    auto log_coeff = -0.5 * (D * std::log(2 * M_PI) + log_det[k]);
                    log_resp.select(1, k) = torch::log(pi[k]) + log_coeff - 0.5 * sq_mahalanobis;
                }
                
                // Normalize responsibilities
                resp = torch::exp(log_resp - torch::logsumexp(log_resp, 1, true));
                
                // M-step: update parameters
                auto Nk = resp.sum(0);
                auto mu_new = resp.t().mm(x).div(Nk.unsqueeze(1));
                auto sigma_new = torch::empty_like(sigma);
                
                // Update mixture weights
                pi = (Nk + alpha - 1) / (N + K * alpha - K);
                
                // Update covariances
                for(int k = 0; k < K; k++) {
                    auto diff_k = x - mu_new[k];
                    auto weighted_diff = diff_k * resp.select(1, k).unsqueeze(1);
                    sigma_new[k] = diff_k.t().mm(weighted_diff) / Nk[k];
                    sigma_new[k].add_(torch::eye(D, options), 1e-4);  // Add regularization
                }
                
                // Check convergence
                auto delta = (mu - mu_new).norm() + (sigma - sigma_new).norm();
                mu = mu_new;
                sigma = sigma_new;
                
                if (delta.item<float>() < tol) {
                    break;
                }
            }
            
            // Get active components
            auto Nk = resp.sum(0);
            
            // Compute final labels
            auto final_labels = torch::empty({N}, idx_options);
            const int batch_size = 10000;
            
            for (int i = 0; i < N; i += batch_size) {
                int current_size = std::min(batch_size, N - i);
                auto batch_x = x.slice(0, i, i + current_size);
                auto batch_log_prob = torch::empty({current_size, K}, options);
                
                for(int k = 0; k < K; k++) {
                    auto solved = at::linalg_solve_triangular(
                        L_batch[k], (batch_x - mu[k]).t(),
                        /*upper=*/false, /*transpose=*/true, /*unitriangular=*/false
                    );
                    auto sq_mahalanobis = solved.t().pow(2).sum(1);
                    auto log_coeff = -0.5 * (D * std::log(2 * M_PI) + 2 * L_batch[k].diagonal(0, -2, -1).log().sum(-1));
                    batch_log_prob.select(1, k) = torch::log(pi[k]) + log_coeff - 0.5 * sq_mahalanobis;
                }
                
                final_labels.slice(0, i, i + current_size) = torch::argmax(batch_log_prob, 1);
            }
            auto active_indices = std::get<0>(at::_unique(final_labels, true));
            pi = pi.index_select(0, active_indices);
            mu = mu.index_select(0, active_indices);
            sigma = sigma.index_select(0, active_indices);
            return std::make_tuple(pi, mu, sigma, final_labels);
        } catch (const std::exception& e) {
            std::cerr << "Error in gem_core: " << e.what() << std::endl;
            throw;
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
    vbem_core(
        torch::Tensor x,
        int K,
        float alpha,
        float tol,
        int max_iter
    ) {
        torch::NoGradGuard no_grad;
        try {
            const int N = x.size(0);
            const int D = x.size(1);
            const auto device = x.device();
            const auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
            const auto idx_options = torch::TensorOptions().device(device).dtype(torch::kInt64);

            // Adjust K if necessary
            K = std::min(K, N);

            // Hyperparameters for NIW prior
            float beta_0 = 1.0;  // Initial sample size
            float nu_0 = D + 2.0;  // Initial degrees of freedom
            auto m_0 = x.mean(0);  // Prior mean
            auto W_0 = torch::eye(D, options);  // Prior scale matrix
            
            // Initialize variational parameters
            auto beta = torch::full({K}, beta_0, options);
            auto nu = torch::full({K}, nu_0, options);
            
            // Initialize means by randomly selecting K points
            auto indices = torch::randperm(N, idx_options).slice(0, 0, K);
            auto m = x.index_select(0, indices);
            
            // Initialize W (precision matrices)
            auto W = W_0.unsqueeze(0).repeat({K, 1, 1});
            
            // Initialize Dirichlet parameters
            auto alpha_k = torch::full({K}, alpha, options);
            
            // Pre-allocate tensors
            auto resp = torch::empty({N, K}, options);
            auto log_resp = torch::empty({N, K}, options);
            auto prev_lb = -std::numeric_limits<float>::infinity();
            
            for(int iter = 0; iter < max_iter; iter++) {
                // E-step: compute responsibilities
                auto log_pi_tilde = torch::digamma(alpha_k) - torch::digamma(alpha_k.sum());
                auto log_det_W = torch::zeros({K}, options);
                
                for(int k = 0; k < K; k++) {
                    log_det_W[k] = torch::logdet(W[k]);
                }
                
                auto E_log_det_Lambda = D * std::log(2.0) + log_det_W;
                for(int d = 0; d < D; d++) {
                    E_log_det_Lambda += torch::digamma((nu - d) / 2.0);
                }
                
                for(int k = 0; k < K; k++) {
                    auto diff = x - m[k];
                    auto E_quad = D / beta[k] + nu[k] * (diff.mm(W[k]).mul(diff)).sum(1);
                    log_resp.select(1, k) = log_pi_tilde[k] + 0.5 * (E_log_det_Lambda[k] - D * std::log(2 * M_PI) - E_quad);
                }
                
                // Normalize responsibilities
                resp = torch::exp(log_resp - torch::logsumexp(log_resp, 1, true));
                
                // M-step: update variational parameters
                auto Nk = resp.sum(0);
                auto x_bar = resp.t().mm(x).div(Nk.unsqueeze(1));
                
                // Update parameters
                for(int k = 0; k < K; k++) {
                    // Update beta
                    beta[k] = beta_0 + Nk[k];
                    
                    // Update nu
                    nu[k] = nu_0 + Nk[k];
                    
                    // Update m
                    m[k] = (beta_0 * m_0 + Nk[k] * x_bar[k]) / beta[k];
                    
                    // Update W
                    auto diff = x - x_bar[k];
                    auto S = diff.t().mm(diff.mul(resp.select(1, k).unsqueeze(1))) / Nk[k];
                    auto diff_means = x_bar[k] - m_0;
                    W[k] = W_0 + Nk[k] * S + (beta_0 * Nk[k] / beta[k]) * diff_means.outer(diff_means);
                }
                
                // Update Dirichlet parameters
                alpha_k = alpha + Nk;
                
                // Compute lower bound for convergence check
                auto lb = torch::sum(resp.mul(log_resp));  // E[log p(Z|pi)]
                if (std::abs(lb.item<float>() - prev_lb) < tol) {
                    break;
                }
                prev_lb = lb.item<float>();
            }
            
            // Compute final parameters
            auto pi = alpha_k / alpha_k.sum();
            auto sigma = torch::empty({K, D, D}, options);
            for(int k = 0; k < K; k++) {
                sigma[k] = W[k].inverse() / (nu[k] - D - 1);
            }
            
            // Compute final labels
            auto final_labels = torch::argmax(resp, 1);
            
            // Get active components
            auto active_indices = std::get<0>(at::_unique(final_labels, true));
            pi = pi.index_select(0, active_indices);
            m = m.index_select(0, active_indices);
            sigma = sigma.index_select(0, active_indices);
            
            return std::make_tuple(pi, m, sigma, final_labels);
            
        } catch (const std::exception& e) {
            std::cerr << "Error in vbem_core: " << e.what() << std::endl;
            throw;
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
    cem_core(
        torch::Tensor x,
        int K,
        float alpha,
        float tol,
        int max_iter
    ) {
        torch::NoGradGuard no_grad;
        try {
            const int N = x.size(0);
            const int D = x.size(1);
            const auto device = x.device();
            const auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
            const auto idx_options = torch::TensorOptions().device(device).dtype(torch::kInt64);

            // Increase regularization constant
            const float reg_const = 1e-3;  // Added regularization constant

            // Adjust K if necessary
            K = std::min(K, N);

            // Initialize model parameters (same as GEM)
            auto pi = torch::full({K}, alpha, options).div(K * alpha);
            auto indices = torch::randperm(N, idx_options).slice(0, 0, K);
            auto mu = x.index_select(0, indices);
            auto sigma = torch::eye(D, options).repeat({K, 1, 1});

            // Pre-allocate tensors
            auto log_resp = torch::empty({N, K}, options);
            auto L_batch = torch::empty({K, D, D}, options);
            auto log_det = torch::empty({K}, options);
            auto prev_labels = torch::empty({N}, idx_options);
            
            for(int iter = 0; iter < max_iter; iter++) {
                // Before Cholesky decomposition, ensure matrix is positive definite
                auto sigma_reg = sigma.clone();
                for(int k = 0; k < K; k++) {
                    // Add regularization to diagonal
                    sigma_reg[k].add_(torch::eye(D, options).mul(reg_const));
                    
                    // Ensure symmetry
                    sigma_reg[k] = (sigma_reg[k] + sigma_reg[k].transpose(-2, -1)) / 2.0;
                    
                    // Add minimum eigenvalue regularization if needed
                    auto eigh_result = torch::linalg::eigh(sigma_reg[k], "L");
                    auto eigenvalues = std::get<0>(eigh_result);
                    auto min_eig = eigenvalues[0].item<float>();
                    if (min_eig < reg_const) {
                        sigma_reg[k].add_(torch::eye(D, options).mul(reg_const - min_eig + 1e-6));
                    }
                }

                // Use regularized covariance for Cholesky
                L_batch = torch::linalg::cholesky((sigma_reg + sigma_reg.transpose(-2, -1))/2);
                log_det = 2 * L_batch.diagonal(0, -2, -1).log().sum(-1);
                auto diff = x.unsqueeze(1) - mu.unsqueeze(0);

                // E-step: compute log probabilities
                for(int k = 0; k < K; k++) {
                    auto solved = at::linalg_solve_triangular(
                        L_batch[k], diff.select(1, k).t(),
                        /*upper=*/false, /*transpose=*/true, /*unitriangular=*/false
                    );
                    auto sq_mahalanobis = solved.t().pow(2).sum(1);
                    auto log_coeff = -0.5 * (D * std::log(2 * M_PI) + log_det[k]);
                    log_resp.select(1, k) = torch::log(pi[k]) + log_coeff - 0.5 * sq_mahalanobis;
                }

                // C-step: hard assignment
                auto labels = torch::argmax(log_resp, 1);

                // Check convergence on labels
                if (iter > 0 && (labels == prev_labels).all().item<bool>()) {
                    break;
                }
                prev_labels = labels;

                // M-step: update parameters
                auto Nk = torch::zeros({K}, options);
                auto mu_new = torch::zeros({K, D}, options);
                auto sigma_new = torch::zeros({K, D, D}, options);

                // Compute cluster statistics
                for(int k = 0; k < K; k++) {
                    auto mask = labels == k;
                    Nk[k] = mask.sum();

                    if (Nk[k].item<float>() > 0) {
                        // Update mean
                        auto cluster_points = x.index({mask});
                        mu_new[k] = cluster_points.mean(0);

                        // Update covariance with regularization
                        auto diff_k = cluster_points - mu_new[k];
                        sigma_new[k] = diff_k.t().mm(diff_k) / Nk[k];
                        sigma_new[k].add_(torch::eye(D, options).mul(reg_const));  // Add regularization
                    } else {
                        // If cluster is empty, keep previous parameters
                        mu_new[k] = mu[k];
                        sigma_new[k] = sigma[k];
                    }
                }

                // Update mixture weights with Dirichlet prior
                pi = (Nk + alpha - 1) / (N + K * alpha - K);

                // Update parameters
                mu = mu_new;
                sigma = sigma_new;
            }
            
            // Get active components
            auto active_indices = std::get<0>(at::_unique(prev_labels, true));
            pi = pi.index_select(0, active_indices);
            mu = mu.index_select(0, active_indices);
            sigma = sigma.index_select(0, active_indices);
            
            return std::make_tuple(pi, mu, sigma, prev_labels);
        } catch (const std::exception& e) {
            std::cerr << "Error in cem_core: " << e.what() << std::endl;
            throw;
        }
    }

    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> 
    gmm_hierarchical_implementation(
        torch::Tensor x,
        torch::Tensor k_list,
        float alpha,
        float tol,
        int max_iter,
        GMMVariant variant
    ) {
        torch::NoGradGuard no_grad;
        const auto device = x.device();
        const auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
        const int depth = k_list.size(0);
        std::vector<torch::Tensor> all_depth_label;
        std::vector<torch::Tensor> all_depth_pi;
        std::vector<torch::Tensor> all_depth_mu;
        std::vector<torch::Tensor> all_depth_sigma;

        // Function pointer to selected GMM variant
        auto gmm_core = [variant](torch::Tensor x, int K, float alpha, float tol, int max_iter) {
            switch (variant) {
                case GMMVariant::GEM:
                    return gem_core(x, K, alpha, tol, max_iter);
                case GMMVariant::VBEM:
                    return vbem_core(x, K, alpha, tol, max_iter);
                case GMMVariant::CEM:
                    return cem_core(x, K, alpha, tol, max_iter);
                default:
                    throw std::runtime_error("Unknown GMM variant");
            }
        };

        // First level clustering using selected variant
        auto [pi, mu, sigma, label] = gmm_core(x, k_list[0].item<int>(), alpha, tol, max_iter);
        auto label_inverse = std::get<1>(at::_unique(label, true, /*return_inverse=*/true));
        all_depth_label.push_back(label_inverse);
        all_depth_pi.push_back(pi);
        all_depth_mu.push_back(mu);
        all_depth_sigma.push_back(sigma);
        
        // Keep track of parent labels for proper hierarchy
        auto current_labels = label_inverse;
        // For each subsequent level
        for(int level = 1; level < depth; level++) {
            auto unique_labels = std::get<0>(at::_unique(current_labels, true));
            auto inside_loop_size = unique_labels.size(0);
            int k = k_list[level].item<int>();
            auto sub_labels = torch::empty_like(current_labels);
            auto sub_pi = std::vector<torch::Tensor>(inside_loop_size);
            auto sub_mu = std::vector<torch::Tensor>(inside_loop_size);
            auto sub_sigma = std::vector<torch::Tensor>(inside_loop_size);
            
            #pragma omp parallel for
            for(int i = 0; i < inside_loop_size; i++) {
                auto mask = current_labels == unique_labels[i];
                if (!mask.any().item<bool>()) {
                    continue;
                }
                auto [pi, mu, sigma, lbl] = gmm_core(
                    x.index({mask}), k, alpha, tol, max_iter
                );
                
                // Each thread writes to its own index i
                sub_labels.index_put_({mask}, lbl + i * k);
                sub_pi[i] = pi;
                sub_mu[i] = mu; 
                sub_sigma[i] = sigma;
            }
            auto level_label = torch::cat(sub_labels);
            auto level_pi = torch::cat(sub_pi);
            auto level_mu = torch::cat(sub_mu);
            auto level_sigma = torch::cat(sub_sigma);
            auto level_label_inverse = std::get<1>(at::_unique(level_label, true, /*return_inverse=*/true));
            all_depth_pi.push_back(level_pi);
            all_depth_mu.push_back(level_mu);
            all_depth_sigma.push_back(level_sigma);
            all_depth_label.push_back(level_label_inverse);
            current_labels = level_label_inverse;
        }
        return std::make_tuple(all_depth_label, all_depth_pi, all_depth_mu, all_depth_sigma);
    }
}

