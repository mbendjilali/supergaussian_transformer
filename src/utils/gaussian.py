from gaussian_mixture_cpp import hierarchical_gmm as gmm
from gaussian_mixture_cpp import GMMVariant

def bayesian_gaussian_mixture_model(x, K, alpha=1.0, tol=1e-2, max_iter=10, variant="CEM"):
    """
    Fit a Bayesian Gaussian Mixture Model to point cloud data using C++ implementation
    Args:
        x (torch.Tensor): Input points of shape (N, D)
        K (int): Number of Gaussian components
        alpha (float): Dirichlet concentration parameter
        tol (float): Convergence tolerance
        max_iter (int): Maximum number of iterations
    Returns:
        tuple: (pi, mu, sigma, cluster) - mixing coefficients, means, covariances, and cluster assignments
    """
    gmm_variant = getattr(GMMVariant, variant)
    return gmm(x, K, alpha, tol, max_iter, gmm_variant)