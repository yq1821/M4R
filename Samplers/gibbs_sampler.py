import numpy as np
from scipy.stats import multivariate_normal, invwishart, dirichlet
from tqdm import tqdm

def gibbs_sampler_gmm_multivariate(X, K, num_iterations):
    """
    Gibbs sampler for a Multivariate Gaussian Mixture Model (GMM).

    Parameters:
        X: array-like, shape (N, p)
            Observed data points (N samples, p dimensions).
        K: int
            Number of clusters.
        num_iterations: int
            Number of iterations for the Gibbs sampler.

    Returns:
        samples: list
            A list of sampled parameters (pi, mu, Sigma, z) for each iteration.
    """
    N, p = X.shape  # Number of data points (N) and dimensions (p)

    # --- Initialization ---
    pi = dirichlet.rvs(np.ones(K) * 2, size=1).flatten()  # Randomized mixing proportions
    mu = np.random.multivariate_normal(np.mean(X, axis=0), np.cov(X.T), size=K)  # Randomized means
    Sigma = np.array([np.cov(X.T) + 1e-6 * np.eye(p) for _ in range(K)])  # Positive-definite covariances

    # Hyperparameters for priors
    alpha = np.ones(K) * 5  # Dirichlet prior for mixing proportions
    mu0 = np.mean(X, axis=0)  # Prior for means
    lambda0 = 1.0  # Precision for means
    nu0 = p + 2  # Degrees of freedom for Inverse-Wishart prior
    Psi0 = np.cov(X.T) * 2  # Scale matrix for Inverse-Wishart prior

    samples = []  # List to store the sampled parameters

    # --- Gibbs Sampling Loop ---
    for iteration in tqdm(range(num_iterations), desc="Sampling"):
        # Step 1: Sample z_i (Cluster Assignments)
        log_posterior = np.log(np.clip(pi, 1e-10, None)) + np.array(
            [multivariate_normal.logpdf(X, mean=mu[k], cov=Sigma[k]) for k in range(K)]
        ).T
        posterior_probs = np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True))  # Numerical stability
        posterior_probs /= posterior_probs.sum(axis=1, keepdims=True)
        z = np.array([np.random.choice(K, p=p) for p in posterior_probs])

        # Step 2: Update mu_k (Cluster Means)
        for k in range(K):
            X_k = X[np.array(z) == k]  # Data points in cluster k
            n_k = len(X_k)
            if n_k > 0:
                mean_k = np.mean(X_k, axis=0)
                mu_n = (lambda0 * mu0 + n_k * mean_k) / (lambda0 + n_k)
                lambda_n = lambda0 + n_k
                Sigma_k = Sigma[k] / lambda_n
                mu[k] = np.random.multivariate_normal(mu_n, Sigma_k)
            else:
                mu[k] = np.random.multivariate_normal(mu0, Sigma_[k] / lambda0)

        # Step 3: Update Sigma_k (Cluster Covariances)
        for k in range(K):
            X_k = X[np.array(z) == k]
            n_k = len(X_k)
            if n_k > 0:
                S_k = np.cov(X_k.T, bias=True) * n_k
                nu_n = nu0 + n_k
                Psi_n = Psi0 + S_k
                Sigma[k] = invwishart.rvs(df=nu_n, scale=Psi_n)
            else:
                Sigma[k] = invwishart.rvs(df=nu0, scale=Psi0)
  
        # Step 4: Update pi (Mixing Proportions)
        counts = np.array([np.sum(np.array(z) == k) for k in range(K)])
        counts[counts == 0] = 1
        pi = dirichlet.rvs(alpha + counts)[0]

        # Store the current samples
        samples.append((pi.copy(), mu.copy(), Sigma.copy(), np.array(z).copy()))
    burn_in = 0
    samples = samples[burn_in:]

    return samples


def bayesian_repulsive(X, K, num_iterations, h):
    """
    Optimized Gibbs Sampler with Bayesian Repulsion.
    """
    N, p = X.shape  # Number of data points (N) and dimensions (p)

    # --- Initialization ---
    pi = dirichlet.rvs(np.ones(K) * 2, size=1).flatten()  # Randomized mixing proportions
    mu = np.random.multivariate_normal(np.mean(X, axis=0), np.cov(X.T), size=K)  # Randomized means
    Sigma = np.array([np.cov(X.T) + 1e-6 * np.eye(p) for _ in range(K)])  # Positive-definite covariances

    # Hyperparameters for priors
    alpha = np.ones(K) * 5  # Dirichlet prior for mixing proportions
    mu0 = np.mean(X, axis=0)  # Prior for means
    lambda0 = 1.0  # Precision for means
    nu0 = p + 2  # Degrees of freedom for Inverse-Wishart prior
    Psi0 = np.cov(X.T) * 2  # Scale matrix for Inverse-Wishart prior

    samples = []  # List to store the sampled parameters

    # --- Gibbs Sampling Loop ---
    for iteration in tqdm(range(num_iterations), desc="Sampling"):
        # Step 1: Sample z_i (Cluster Assignments)
        log_posterior = np.log(np.clip(pi, 1e-10, None)) + np.array(
            [multivariate_normal.logpdf(X, mean=mu[k], cov=Sigma[k]) for k in range(K)]
        ).T
        posterior_probs = np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True))  # Numerical stability
        posterior_probs /= posterior_probs.sum(axis=1, keepdims=True)
        z = np.array([np.random.choice(K, p=p) for p in posterior_probs])

        # Step 2: Update mu_k (Cluster Means) with M-H
        for k in range(K):
            X_k = X[z == k]
            n_k = len(X_k)
            if n_k > 0:
                mean_k = np.mean(X_k, axis=0)
                mu_n = (lambda0 * mu0 + n_k * mean_k) / (lambda0 + n_k)
                lambda_n = lambda0 + n_k
                Sigma_k = Sigma[k] / lambda_n

                # Propose new mu_k
                mu_proposed = np.random.multivariate_normal(mu_n, Sigma_k)
                current_h = h(mu)
                proposed_mu = mu.copy()
                proposed_mu[k] = mu_proposed
                proposed_h = h(proposed_mu)

                log_acceptance_ratio = (
                    multivariate_normal.logpdf(mu_proposed, mean=mu_n, cov=Sigma_k)
                    - multivariate_normal.logpdf(mu[k], mean=mu_n, cov=Sigma_k)
                    + np.log(np.maximum(proposed_h, 1e-10))
                    - np.log(np.maximum(current_h, 1e-10))
                )
                if np.log(np.random.rand()) < log_acceptance_ratio:
                    mu[k] = mu_proposed  # Accept

        # Step 3: Update Sigma_k (Cluster Covariances)
        for k in range(K):
            X_k = X[z == k]
            n_k = len(X_k)
            if n_k > 0:
                S_k = np.cov(X_k.T, bias=True) * n_k
                nu_n = nu0 + n_k
                Psi_n = Psi0 + S_k
                Sigma[k] = invwishart.rvs(df=nu_n, scale=Psi_n)
            else:
                Sigma[k] = invwishart.rvs(df=nu0, scale=Psi0)

        # Step 4: Update pi (Mixing Proportions)
        counts = np.bincount(z, minlength=K)  # Count points in each cluster
        pi = dirichlet.rvs(alpha + counts)[0]  # Sample from the Dirichlet distribution

        # Store the current samples
        samples.append((pi.copy(), mu.copy(), Sigma.copy(), z.copy()))

    # Burn-in period
    burn_in = 100
    samples = samples[burn_in:]

    return samples

def plot_with_reference_lines(mu_samples, sigma_samples, bmi_bounds, sbp_bounds, ax):
    """
    Plots clusters with posterior mean and variance, and adds reference lines for h2.
    """
    posterior_mu = np.mean(mu_samples, axis=0)
    posterior_sigma = np.mean(sigma_samples, axis=0)

    # Scatter data points
    ax.scatter(bmi_sbp_data['bmi'], bmi_sbp_data['sbp'], alpha=0.7, color='grey', label='Data Points')

    # Plot each cluster's posterior mean and confidence ellipse
    for k in range(len(posterior_mu)):
        # Plot the posterior mean of the cluster
        ax.scatter(
            posterior_mu[k, 0],
            posterior_mu[k, 1],
            color='black',
            s=100,
            label=f"Cluster {k + 1} Mean" if k == 0 else None,
            marker='x'
        )

        # Add confidence ellipse using posterior covariance
        cov_matrix = posterior_sigma[k]
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        angle = np.arctan2(*eigenvectors[:, 0][::-1])
        width, height = 2 * np.sqrt(eigenvalues)  # 1 SD ellipse
        ellipse = patches.Ellipse(
            posterior_mu[k], width, height, angle=np.degrees(angle),
            edgecolor='black', facecolor='none', linestyle='--'
        )
        ax.add_patch(ellipse)

    # Add BMI and SBP reference lines
    for bmi_bound in bmi_bounds:
        ax.axvline(x=bmi_bound, color='red', linestyle='--', linewidth=1, label='BMI Boundaries' if bmi_bound == bmi_bounds[0] else None)

    for sbp_bound in sbp_bounds:
        ax.axhline(y=sbp_bound, color='blue', linestyle='--', linewidth=1, label='SBP Boundaries' if sbp_bound == sbp_bounds[0] else None)

    ax.set_xlabel("BMI")
    ax.set_ylabel("SBP")
    ax.grid()
    ax.legend()
