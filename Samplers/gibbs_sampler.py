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
    Sigma = np.array([np.cov(X.T) for _ in range(K)]) 

    # Hyperparameters for priors
    alpha = np.ones(K) * 5  # Dirichlet prior for mixing proportions
    mu0 = np.mean(X, axis=0)  # Prior for means
    V0 = 1.0  # Precision for means
    nu0 = p + 2  # Degrees of freedom for Inverse-Wishart prior
    S0 = np.cov(X.T) * 2  # Scale matrix for Inverse-Wishart prior
    

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
                S_n = S0 + S_k
                Sigma[k] = invwishart.rvs(df=nu_n, scale=S_n)
            else:
                Sigma[k] = invwishart.rvs(df=nu0, scale=S0)
  
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
    Gibbs Sampler with Bayesian Repulsion.
    """
    N, p = X.shape  # Number of data points (N) and dimensions (p)

    # --- Initialization ---
    pi = dirichlet.rvs(np.ones(K) * 2, size=1).flatten()  # Randomized mixing proportions
    mu = np.random.multivariate_normal(np.mean(X, axis=0), np.cov(X.T), size=K)  # Randomized means
    Sigma = np.array([np.cov(X.T) for _ in range(K)]) 

    # Hyperparameters for priors
    alpha = np.ones(K) * 5  # Dirichlet prior for mixing proportions
    mu0 = np.mean(X, axis=0)  # Prior for means
    m0 = np.mean(X, axis=0)  # Prior mean
    V0 = 5*np.eye(p)  # Identity matrix of shape (p, p)
    nu0 = p + 2  # Degrees of freedom for Inverse-Wishart prior
    S0 = np.cov(X.T) * 2  # Scale matrix for Inverse-Wishart prior
    

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
                V_n = np.linalg.inv(np.linalg.inv(V0) + n_k * np.linalg.inv(Sigma[k]))
                m_n = V_n @ (np.linalg.inv(V0) @ mu0 + n_k * np.linalg.inv(Sigma[k]) @ mean_k)
                #propse new mu_k using random walk
                tau = np.std(X, axis=0) * 0.1  # Scale tau relative to data
                M = np.eye(p) * tau**2
                mu_proposed = np.random.multivariate_normal(mu[k], M)
                #compute the acceptance probability
                proposed_mu = mu.copy()
                proposed_mu[k] = mu_proposed
                log_acceptance_rate = (
                    multivariate_normal.logpdf(mu_proposed, mean=m0, cov=V0)
                    + np.log(h(proposed_mu))
                    + np.sum(multivariate_normal.logpdf(X_k, mean=mu_proposed, cov=Sigma[k]))
                ) - (
                    multivariate_normal.logpdf(mu[k], mean=m0, cov=V0)
                    + np.log(h(mu))
                    + np.sum(multivariate_normal.logpdf(X_k, mean=mu[k], cov=Sigma[k]))
                    + 1e-10
                )
                acceptance_rate = np.exp(np.clip(log_acceptance_rate, -100, 0))  # Prevent overflow
                # Accept or reject
                if np.random.rand() < min(1, acceptance_rate):
                    mu[k] = mu_proposed  # Accept the proposed mu

        # Step 3: Update Sigma_k (Cluster Covariances)
        for k in range(K):
            X_k = X[z == k]
            n_k = len(X_k)
            if n_k > 0:
                S_k = np.cov(X_k.T, bias=True) * n_k
                nu_n = nu0 + n_k
                S_n = S0 + S_k
                Sigma[k] = invwishart.rvs(df=nu_n, scale=S_n)
            else:
                Sigma[k] = invwishart.rvs(df=nu0, scale=S0)

        # Step 4: Update pi (Mixing Proportions)
        counts = np.bincount(z, minlength=K)  # Count points in each cluster
        pi = dirichlet.rvs(alpha + counts)[0]  # Sample from the Dirichlet distribution

        # Store the current samples
        samples.append((pi.copy(), mu.copy(), Sigma.copy(), z.copy()))

    # Burn-in period
    burn_in = 100
    samples = samples[burn_in:]

    return samples
