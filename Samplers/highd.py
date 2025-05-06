import numpy as np
from scipy.stats import dirichlet, multivariate_normal, invwishart
from tqdm import tqdm

def gibbs_sampler_gmm_multivariate(X, K, num_iterations, burn_in):
    """
    Conjugate Gibbs Sampler for a Gaussian Mixture Model (GMM), without repulsive priors.

    Parameters:
        X : ndarray of shape (N, p)
            Observed data points.
        K : int
            Number of clusters.
        num_iterations : int
            Total iterations for the Gibbs sampler.

    Returns:
        samples : list
            A list of sampled parameters (pi, mu, Sigma, z) across iterations.
    """
    N, p = X.shape

    cov0 = np.cov(X,rowvar=False)  # Covariance of the data
    cov0 += np.eye(p)*1e-6  # Regularize covariance matrix for numerical stability
    # -- Initialization --
    pi = dirichlet.rvs([2]*K, size=1)[0]  # Mixing proportions
    mu = np.random.multivariate_normal(np.mean(X, axis=0), np.cov(X.T), size=K)
    Sigma = np.array([np.cov(X.T) for _ in range(K)])

    # -- Hyperparameters --
    alpha = np.ones(K) * 0.5     # Dirichlet prior for pi #larger alpha means more uniform distribution
    m0 = np.mean(X, axis=0)     # Prior mean for mu
    V0 = np.eye(p) * 5          # Prior covariance for mu #larger V0, more spread out
    V0_inv = np.linalg.inv(V0)
    nu0 = p + 2                # Deg. of freedom for Inv-Wishart #larger nu0, more concentrated
    S0 = np.cov(X.T) * 1      # Scale matrix for Inv-Wishart #larger S0,more spread out

    samples = []

    # --- Gibbs Sampling Loop ---
    for iteration in tqdm(range(num_iterations), desc="Sampling"):
        # Sample z_i (Cluster Assignments)
        log_posterior = np.log(np.clip(pi, 1e-10, None)) + np.array(
            [multivariate_normal.logpdf(X, mean=mu[k], cov=Sigma[k]) for k in range(K)]
        ).T
        posterior_probs = np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True))  # Numerical stability
        posterior_probs /= posterior_probs.sum(axis=1, keepdims=True)
        z = np.array([np.random.choice(K, p=p) for p in posterior_probs])

        # Sample mu_k from its conjugate posterior
        for k in range(K):
            X_k = X[z == k]
            n_k = len(X_k)
            if n_k > 0:
                xbar_k = X_k.mean(axis=0)
                # Posterior for mu_k:
                # V_k^{-1} = V0^{-1} + n_k * Sigma[k]^{-1}
                Sigma_k_inv = np.linalg.inv(Sigma[k])
                V_k_inv = V0_inv + n_k * Sigma_k_inv
                V_k = np.linalg.inv(V_k_inv)
                m_k = V_k @ (V0_inv @ m0 + n_k * Sigma_k_inv @ xbar_k)
                mu[k] = np.random.multivariate_normal(m_k, V_k)
            else:
                # If no points in cluster k, sample from prior
                mu[k] = np.random.multivariate_normal(m0, V0)

        # Sample Sigma_k from InvWishart posterior
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

        # Sample pi (mixing proportions)
        counts = np.array([np.sum(z == k_) for k_ in range(K)])
        pi = dirichlet.rvs(alpha + counts)[0]

        # Store current iteration's samples
        samples.append((pi.copy(), mu.copy(), Sigma.copy(), z.copy()))
    
    return samples[burn_in:]



def bayesian_repulsive_randomwalk(X, K, num_iterations, h, burn_in, sig):
    """
    Gibbs Sampler with Bayesian Repulsion, using random-walk proposals.

    Parameters:
        X : ndarray of shape (N, p)
            Observed data points.
        K : int
            Number of clusters.
        num_iterations : int
            Total number of iterations.
        h : function
            Repulsive prior function.
        burn_in : int
            Number of initial samples to discard.
        sig : float
            Standard deviation of the random walk proposal.

    Returns:
        samples : list
            A list of sampled parameters (pi, mu, Sigma, z) after burn-in.
    """
    N, p = X.shape  # Number of data points and features

    # --- Initialization ---
    pi = dirichlet.rvs(np.ones(K) * 2, size=1).flatten()  # Mixing proportions

    # Regularize covariance matrix for high-dimensional stability
    cov_X = np.cov(X.T)
    cov_X += np.eye(p) * 1e-6  # Add small jitter for numerical stability

    mu = np.random.multivariate_normal(np.mean(X, axis=0), cov_X, size=K)  # Randomized initial means
    Sigma = np.array([cov_X for _ in range(K)])  # Initialize cluster covariances

    # --- Hyperparameters ---
    alpha = np.ones(K) * 5    # Dirichlet prior for pi
    mu0 = np.mean(X, axis=0)  # Prior mean for mu
    V0 = 5 * np.eye(p)        # Prior covariance for mu
    nu0 = p + 2               # Degrees of freedom for Inverse-Wishart
    S0 = np.cov(X.T) * 2      # Scale matrix for Inverse-Wishart

    samples = []  # To store sampled parameters

    # --- Gibbs Sampling Loop ---
    for iteration in tqdm(range(num_iterations), desc="Sampling"):
        # Step 1: Sample z_i (Cluster Assignments)
        log_posterior = np.log(np.clip(pi, 1e-10, None)) + np.array(
            [multivariate_normal.logpdf(X, mean=mu[k], cov=Sigma[k]) for k in range(K)]
        ).T
        posterior_probs = np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True))  # Numerical stability
        posterior_probs /= posterior_probs.sum(axis=1, keepdims=True)
        z = np.array([np.random.choice(K, p=p) for p in posterior_probs])

        # Step 2: Update mu_k (Cluster Means) using Random Walk Metropolis-Hastings
        for k in range(K):
            X_k = X[z == k]  # Data points assigned to cluster k
            n_k = len(X_k)
            if n_k > 0:
                # Propose new mu_k
                mu_proposed = np.random.multivariate_normal(mu[k], sig**2 * np.eye(p))
                
                # Create a proposed mu array
                proposed_mu = mu.copy()
                proposed_mu[k] = mu_proposed

                # Compute log acceptance ratio
                log_acceptance_rate = (
                    multivariate_normal.logpdf(mu_proposed, mean=mu0, cov=V0)
                    + np.log(h(proposed_mu))
                    + np.sum(multivariate_normal.logpdf(X_k, mean=mu_proposed, cov=Sigma[k]))
                ) - (
                    multivariate_normal.logpdf(mu[k], mean=mu0, cov=V0)
                    + np.log(h(mu))
                    + np.sum(multivariate_normal.logpdf(X_k, mean=mu[k], cov=Sigma[k]))
                )

                # Metropolis-Hastings acceptance
                if np.log(np.random.rand()) < log_acceptance_rate:
                    mu[k] = mu_proposed  # Accept move
            else:
                # If no points in cluster, sample from prior
                mu[k] = np.random.multivariate_normal(mu0, V0)

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
        counts = np.bincount(z, minlength=K)
        pi = dirichlet.rvs(alpha + counts)[0]

        # Store the current samples
        samples.append((pi.copy(), mu.copy(), Sigma.copy(), z.copy()))

    return samples[burn_in:]
