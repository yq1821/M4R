
def gibbs_sampler_gmm_multivariate(X, K, num_iterations):
    import numpy as np
    from scipy.stats import multivariate_normal, invwishart, dirichlet
    from tqdm import tqdm

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
    # z = np.random.randint(0, K, size=N)  # Random initial cluster assignments
    z = np.random.choice(K, size=N)  # Random initial cluster assignments
    pi = np.ones(K) / K  # Uniform weights
    mu = np.full((K, p), np.mean(X, axis=0))  # Initialize with the global mean
    Sigma = np.array([np.cov(X.T) for _ in range(K)])  # Initialize with the global covariance

    # Hyperparameters for priors
    alpha = np.ones(K) * 5  # Dirichlet prior for mixing proportions
    mu0 = np.mean(X, axis=0)  # Prior for means
    lambda0 = 1.0  # Precision for means
    nu0 = p + 2  # Degrees of freedom for Inverse-Wishart prior
    Psi0 = np.cov(X.T) * 2  # Scale matrix for Inverse-Wishart prior

    samples = []  # List to store the sampled parameters

    # --- Gibbs Sampling Loop ---
    for _ in tqdm(range(num_iterations), desc="Sampling"):
        # Step 1: Sample z_i (Cluster Assignments)
        log_posterior = np.zeros((N, K))
        for k in range(K):
            log_posterior[:, k] = np.log(pi[k]) + multivariate_normal.logpdf(X, mean=mu[k], cov=Sigma[k])
        posterior_probs = np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True)) # Use log for stability
        posterior_probs /= posterior_probs.sum(axis=1, keepdims=True)
        z = [np.random.choice(K, p=posterior_probs[i]) for i in range(N)]

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
        print(Sigma)

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
    import numpy as np
    from scipy.stats import multivariate_normal, invwishart, dirichlet
    from tqdm import tqdm

    N, p = X.shape  # Number of data points (N) and dimensions (p)

    # --- Initialization ---
    z = np.random.randint(0, K, size=N)  # Random initial cluster assignments
    pi = np.ones(K) / K  # Uniform weights
    mu = np.full((K, p), np.mean(X, axis=0))   # Initialize means
    Sigma = np.array([np.cov(X.T) for _ in range(K)])  # Initialize covariances

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
        log_posterior = np.zeros((N, K))
        for k in range(K):
            log_posterior[:, k] = np.log(np.maximum(pi[k], 1e-10)) + multivariate_normal.logpdf(X, mean=mu[k], cov=Sigma[k])
        log_max = log_posterior.max(axis=1, keepdims=True)
        log_posterior -= log_max  # Normalize for numerical stability
        posterior_probs = np.exp(log_posterior)
        posterior_probs /= posterior_probs.sum(axis=1, keepdims=True)
        z = np.array([np.random.choice(K, p=posterior_probs[i]) for i in range(N)])

        # Step 2: Update mu_k (Cluster Means) with M-H
        for k in range(K):
            X_k = X[z == k]  # Data points in cluster k
            n_k = len(X_k)
            if n_k > 0:
                # Compute posterior mean and covariance
                mean_k = np.mean(X_k, axis=0)
                mu_n = (lambda0 * mu0 + n_k * mean_k) / (lambda0 + n_k)
                lambda_n = lambda0 + n_k
                Sigma_k = Sigma[k] / lambda_n

                # Propose new mu_k
                mu_proposed = np.random.multivariate_normal(mu_n, Sigma_k)

                # Compute the acceptance probability
                current_h = h(mu)
                proposed_mu = mu.copy()
                proposed_mu[k] = mu_proposed
                proposed_h = h(proposed_mu)

                # Log acceptance ratio
                log_acceptance_ratio = (
                    multivariate_normal.logpdf(mu_proposed, mean=mu_n, cov=Sigma_k)
                    - multivariate_normal.logpdf(mu[k], mean=mu_n, cov=Sigma_k)
                    + np.log(np.maximum(proposed_h, 1e-10))  # h for proposed
                    - np.log(np.maximum(current_h, 1e-10))  # h for current
                )

                # Accept or reject
                if 0.5 < log_acceptance_ratio:
                    mu[k] = mu_proposed  # Accept proposal
            else:
                # If no points in cluster, revert to the prior
                mu[k] = np.random.multivariate_normal(mu0, Sigma[k] / lambda0)

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
        counts = np.array([np.sum(z == k) for k in range(K)])
        counts[counts == 0] = 1
        pi = dirichlet.rvs(alpha + counts)[0]

        # Store the current samples
        samples.append((pi.copy(), mu.copy(), Sigma.copy(), z.copy()))

    burn_in = 100  # Optional burn-in period
    samples = samples[burn_in:]

    return samples

def bayesian_repulsive_with_variance_constraint(X, K, num_iterations, h):
    import numpy as np
    from scipy.stats import multivariate_normal, invwishart, dirichlet
    from tqdm import tqdm

    N, p = X.shape  # Number of data points (N) and dimensions (p)

    # --- Initialization ---
    z = np.random.randint(0, K, size=N)  # Random initial cluster assignments
    pi = np.ones(K) / K  # Uniform weights
    mu = np.full((K, p), np.mean(X, axis=0))  # Initialize means
    Sigma = np.array([np.cov(X.T) for _ in range(K)])  # Initialize covariances

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
        log_posterior = np.zeros((N, K))
        for k in range(K):
            log_posterior[:, k] = np.log(np.maximum(pi[k], 1e-10)) + multivariate_normal.logpdf(X, mean=mu[k], cov=Sigma[k])
        log_max = log_posterior.max(axis=1, keepdims=True)
        log_posterior -= log_max  # Normalize for numerical stability
        posterior_probs = np.exp(log_posterior)
        posterior_probs /= posterior_probs.sum(axis=1, keepdims=True)
        z = np.array([np.random.choice(K, p=posterior_probs[i]) for i in range(N)])

        # Step 2: Update mu_k (Cluster Means) with M-H
        for k in range(K):
            X_k = X[z == k]  # Data points in cluster k
            n_k = len(X_k)
            if n_k > 0:
                # Compute posterior mean and covariance
                mean_k = np.mean(X_k, axis=0)
                mu_n = (lambda0 * mu0 + n_k * mean_k) / (lambda0 + n_k)
                lambda_n = lambda0 + n_k
                Sigma_k = Sigma[k] / lambda_n

                # Propose new mu_k
                mu_proposed = np.random.multivariate_normal(mu_n, Sigma_k)

                # Compute the acceptance probability
                current_h = h(mu)
                proposed_mu = mu.copy()
                proposed_mu[k] = mu_proposed
                proposed_h = h(proposed_mu)

                # Log acceptance ratio
                log_acceptance_ratio = (
                    multivariate_normal.logpdf(mu_proposed, mean=mu_n, cov=Sigma_k)
                    - multivariate_normal.logpdf(mu[k], mean=mu_n, cov=Sigma_k)
                    + np.log(np.maximum(proposed_h, 1e-10))  # h for proposed
                    - np.log(np.maximum(current_h, 1e-10))  # h for current
                )

                # Accept or reject
                if np.log(np.random.rand()) < log_acceptance_ratio:
                    mu[k] = mu_proposed  # Accept proposal
            else:
                # If no points in cluster, revert to the prior
                mu[k] = np.random.multivariate_normal(mu0, Sigma[k] / lambda0)

# Step 3: Update Sigma_k (Cluster Covariances) with Eigenvalue Constraints
        for k in range(K):
            X_k = X[z == k]
            n_k = len(X_k)
            if n_k > 0:
                # Compute posterior parameters
                S_k = np.cov(X_k.T, bias=True) * n_k
                nu_n = nu0 + n_k
                Psi_n = Psi0 + S_k
                
                # Sample a proposed covariance matrix
                proposed_sigma = invwishart.rvs(df=nu_n, scale=Psi_n)
                
                # Compute eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eigh(proposed_sigma)
                
                # Clamp eigenvalues to the range [50, 300]
                constrained_eigenvalues = np.clip(eigenvalues, 30, 300)
                
                # Reconstruct the covariance matrix
                constrained_sigma = eigenvectors @ np.diag(constrained_eigenvalues) @ eigenvectors.T
                
                # Update Sigma_k
                Sigma[k] = constrained_sigma
            else:
                # If no points in cluster, sample from prior
                Sigma[k] = invwishart.rvs(df=nu0, scale=Psi0)

        # Step 4: Update pi (Mixing Proportions)
        counts = np.array([np.sum(z == k) for k in range(K)])
        counts[counts == 0] = 1
        pi = dirichlet.rvs(alpha + counts)[0]

        # Store the current samples
        samples.append((pi.copy(), mu.copy(), Sigma.copy(), z.copy()))

    burn_in = 100  # Optional burn-in period
    samples = samples[burn_in:]

    return samples