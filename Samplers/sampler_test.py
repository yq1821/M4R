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

def gibbs_sampler_gmm_multivariate_joint_mu(X, K, num_iterations, burn_in):
    """
    Gibbs sampler for a Gaussian Mixture Model (GMM) with a joint Metropolis–Hastings update
    for the cluster means (mu). Other parameters are updated conditionally.
    
    Parameters:
        X : ndarray of shape (N, p)
            Observed data points.
        K : int
            Number of clusters.
        num_iterations : int
            Total iterations for the sampler.
        burn_in : int
            Number of burn-in iterations to discard.
    
    Returns:
        samples : list
            A list of sampled parameters (pi, mu, Sigma, z) after burn-in.
    """


    N, p = X.shape

    # -- Initialization --
    pi = dirichlet.rvs([2]*K, size=1)[0]  # Mixing proportions
    mu = np.random.multivariate_normal(np.mean(X, axis=0), np.cov(X.T), size=K)
    Sigma = np.array([np.cov(X.T) for _ in range(K)])

    # -- Hyperparameters --
    alpha = np.ones(K) * 2    # Dirichlet prior for pi
    m0 = np.mean(X, axis=0)     # Prior mean for mu
    V0 = np.cov(X, rowvar=False)  # Prior covariance for mu
    V0_inv = np.linalg.inv(V0)
    nu0 = p + 2               # Degrees of freedom for Inv-Wishart
    S0 = np.cov(X.T) * 2      # Scale matrix for Inv-Wishart

    samples = []

    # Tuning parameter for joint random-walk proposal for mu
    proposal_scale = 1

    # --- Gibbs Sampling Loop ---
    for iteration in tqdm(range(num_iterations), desc="Sampling"):
        # --- Step 1: Update cluster assignments (z) ---
        log_posterior = np.log(np.clip(pi, 1e-10, None)) + np.array([
            multivariate_normal.logpdf(X, mean=mu[k], cov=Sigma[k])
            for k in range(K)
        ]).T
        # For numerical stability
        log_posterior = log_posterior - log_posterior.max(axis=1, keepdims=True)
        posterior_probs = np.exp(log_posterior)
        posterior_probs /= posterior_probs.sum(axis=1, keepdims=True)
        z = np.array([np.random.choice(K, p=p) for p in posterior_probs])

        # --- Step 2: Joint update of all mu via Metropolis–Hastings ---
        # Propose a new mu for every cluster
        mu_proposed = mu + np.random.normal(loc=0, scale=proposal_scale, size=mu.shape)
        log_target_current = 0.0
        log_target_proposed = 0.0
        # Compute the joint target density over mu's (prior and likelihood contributions)
        for k in range(K):
            X_k = X[z == k]
            n_k = len(X_k)
            # Likelihood contribution for points in cluster k
            if n_k > 0:
                log_target_current += ( multivariate_normal.logpdf(mu[k], mean=m0, cov=V0)
                                        + np.sum(multivariate_normal.logpdf(X_k, mean=mu[k], cov=Sigma[k]) ) )
                log_target_proposed += ( multivariate_normal.logpdf(mu_proposed[k], mean=m0, cov=V0)
                                         + np.sum(multivariate_normal.logpdf(X_k, mean=mu_proposed[k], cov=Sigma[k]) ) )
            else:
                # If no data in cluster k, only the prior counts.
                log_target_current += multivariate_normal.logpdf(mu[k], mean=m0, cov=V0)
                log_target_proposed += multivariate_normal.logpdf(mu_proposed[k], mean=m0, cov=V0)
        # Note: Since the symmetric Gaussian random-walk proposal cancels out,
        # we simply compute the acceptance ratio.
        acceptance_ratio = np.exp(log_target_proposed - log_target_current)
        if np.random.rand() < min(1, acceptance_ratio):
            mu = mu_proposed.copy()

        # --- Step 3: Update Sigma_k from the Inverse-Wishart posterior ---
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

        # --- Step 4: Update mixing proportions, pi ---
        counts = np.array([np.sum(z == k) for k in range(K)])
        pi = dirichlet.rvs(alpha + counts)[0]

        # --- Store current iteration's samples ---
        samples.append((pi.copy(), mu.copy(), Sigma.copy(), z.copy()))
    
    return samples[burn_in:]

def bayesian_repulsive_randomwalk(X, K, num_iterations, h, burn_in, sig):
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
                #propse new mu_k using random walk
                mu_proposed = np.random.multivariate_normal(mu[k], sig**2*np.eye(p))
                #compute the acceptance probability
                proposed_mu = mu.copy()
                proposed_mu[k] = mu_proposed
                log_acceptance_rate = (
                    multivariate_normal.logpdf(mu_proposed, mean=m0, cov=V0)
                    + np.log(h(proposed_mu))
                    + np.sum(multivariate_normal.logpdf(X_k, mean=mu_proposed, cov=Sigma[k])) #check
                ) - (
                    multivariate_normal.logpdf(mu[k], mean=m0, cov=V0)
                    + np.log(h(mu))
                    + np.sum(multivariate_normal.logpdf(X_k, mean=mu[k], cov=Sigma[k]))
                     # small constant to avoid division by zero
                )
                # acceptance_rate = np.exp(np.clip(log_acceptance_rate, -100, 0))  # Prevent overflow
                if np.log(np.random.rand()) < log_acceptance_rate:
                    mu[k] = mu_proposed
                # # Accept or reject the proposed value for mu_k
                # if np.random.rand() < min(1, np.exp(log_acceptance_rate)):
                #     mu[k] = mu_proposed  # Accept the proposed mu for cluster k

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

    return samples[burn_in:]

def bayesian_repulsive_randomwalk_joint(X, K, num_iterations, h, burn_in):
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
        # --- Joint update of all cluster means (mu) ---
        mu_proposed = np.empty_like(mu)
        for k in range(K):
            mu_proposed[k] = np.random.multivariate_normal(mu[k], np.eye(p)*0.5)

        # Compute the joint target densities (log domain)
        log_target_current = 0.0
        log_target_proposed = 0.0
        for k in range(K):
            X_k = X[np.array(z) == k]
            if len(X_k) > 0:
                log_target_current += (
                    multivariate_normal.logpdf(mu[k], mean=m0, cov=V0)
                    + np.sum(multivariate_normal.logpdf(X_k, mean=mu[k], cov=Sigma[k]))
                )
                log_target_proposed += (
                    multivariate_normal.logpdf(mu_proposed[k], mean=m0, cov=V0)
                    + np.sum(multivariate_normal.logpdf(X_k, mean=mu_proposed[k], cov=Sigma[k]))
                )
            else:
                # If no data points in cluster k, only include the prior for mu[k]
                log_target_current += multivariate_normal.logpdf(mu[k], mean=m0, cov=V0)
                log_target_proposed += multivariate_normal.logpdf(mu_proposed[k], mean=m0, cov=V0)

        # Include the repulsive prior over the entire configuration.
        # Here, h(mu) should be defined so that it returns a value in (0,1].
        log_target_current += np.log(h(mu))
        log_target_proposed += np.log(h(mu_proposed))

        # Since our proposal is symmetric, the proposal density terms cancel.
        log_acceptance_rate = log_target_proposed - log_target_current
        acceptance_rate = np.exp(np.clip(log_acceptance_rate, -100, 0))
        if np.random.rand() < min(1, acceptance_rate):
            mu = mu_proposed.copy()  # Accept the joint proposal

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

    return samples[burn_in:]

def bayesian_repulsive_neighbor(X, K, num_iterations, h, burn_in, bmi_bounds, sbp_bounds):
    """
    Gibbs Sampler with Bayesian Repulsion.
    
    Parameters:
        X : ndarray (N x p)
            Data points.
        K : int
            Number of clusters.
        num_iterations : int
            Total iterations.
        h : function
            Repulsive prior function.
        burn_in : int
            Number of initial samples to discard.
        bmi_bounds : ndarray
            Array of BMI boundaries (e.g., [-1.57149027, -0.71523168, -0.1036184, inf]).
        sbp_bounds : ndarray
            Array of SBP boundaries (e.g., [0.95012032, 2.12990549, inf]).
    
    Returns:
        samples : list
            List of samples after burn-in.
    """
    N, p = X.shape  # p should be 2 for BMI and SBP

    # --- Initialization ---
    pi = dirichlet.rvs(np.ones(K) * 2, size=1).flatten()  # Randomized mixing proportions
    mu = np.random.multivariate_normal(np.mean(X, axis=0), np.cov(X.T), size=K)  # Randomized means
    Sigma = np.array([np.cov(X.T) for _ in range(K)]) 

    # Hyperparameters for priors
    alpha = np.ones(K) * 5  # Dirichlet prior for mixing proportions
    m0 = np.mean(X, axis=0)  # Prior mean for mu
    V0 = 5 * np.eye(p)       # Prior covariance for mu
    nu0 = p + 2              # Degrees of freedom for Inv-Wishart
    S0 = np.cov(X.T) * 2     # Scale matrix for Inv-Wishart

    samples = []  # List to store sampled parameters

    # --- Gibbs Sampling Loop ---
    for iteration in tqdm(range(num_iterations), desc="Sampling"):
        # Step 1: Sample z_i (Cluster Assignments)
        log_posterior = np.log(np.clip(pi, 1e-10, None)) + np.array(
            [multivariate_normal.logpdf(X, mean=mu[k], cov=Sigma[k]) for k in range(K)]
        ).T
        posterior_probs = np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True))
        posterior_probs /= posterior_probs.sum(axis=1, keepdims=True)
        z = np.array([np.random.choice(K, p=p) for p in posterior_probs])

        # Step 2: Update mu_k (Cluster Means) with neighbor block proposal

        for k in range(K):
            X_k = X[np.array(z) == k]  # Data points in cluster k
            n_k = len(X_k)
            if n_k > 0:
                # --- Neighbor Block Proposal  ---
                # Determine the current bin for each dimension
                current_bin_bmi = np.digitize(mu[k, 0], bins=bmi_bounds)
                current_bin_sbp = np.digitize(mu[k, 1], bins=sbp_bounds)
                
                # Exclude the current bin; candidate bins are only the neighbors (left and right)
                candidate_bins_bmi = [i for i in [current_bin_bmi - 1, current_bin_bmi, current_bin_bmi + 1]
                                    if 1 <= i <= len(bmi_bounds) - 1]
                candidate_bins_sbp = [i for i in [current_bin_sbp - 1, current_bin_sbp, current_bin_sbp + 1]
                                    if 1 <= i <= len(sbp_bounds) - 1]
                
                # Randomly choose a candidate bin for each dimension
                [new_bin_bmi, new_bin_sbp] = [np.random.choice(bins) for bins in [candidate_bins_bmi, candidate_bins_sbp]]

                # Determine boundaries for the chosen candidate bins
                lower_bmi = bmi_bounds[new_bin_bmi - 1]
                upper_bmi = bmi_bounds[new_bin_bmi]
                lower_sbp = sbp_bounds[new_bin_sbp - 1]
                upper_sbp = sbp_bounds[new_bin_sbp]

                # Replace infinite bounds with the fixed margin 
                if np.isinf(lower_bmi):
                    lower_bmi = -2
                if np.isinf(upper_bmi):
                    upper_bmi = 6
                if np.isinf(lower_sbp):
                    lower_sbp = -3
                if np.isinf(upper_sbp):
                    upper_sbp = 5

                # Sample uniformly within the candidate neighbor block:
                mu_proposed = np.array([
                    np.random.uniform(lower_bmi, upper_bmi),
                    np.random.uniform(lower_sbp, upper_sbp)
                ])
                # --- End of Proposal ---

                # Compute the acceptance probability
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
                    + 1e-6  # Small constant to avoid division by zero
                )
                acceptance_rate = np.exp(np.clip(log_acceptance_rate, -100, 0))
                if np.random.rand() < min(1, acceptance_rate):
                    mu[k] = mu_proposed  # Accept the proposed mu for cluster k
            else:
                mu[k] = np.random.multivariate_normal(m0, V0)

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
        counts = np.bincount(z, minlength=K)
        pi = dirichlet.rvs(alpha + counts)[0]

        samples.append((pi.copy(), mu.copy(), Sigma.copy(), z.copy()))

    return samples[burn_in:]




def bayesian_repulsive_new(X, K, num_iterations, h, burn_in, bmi_bounds, sbp_bounds):
    """
    Gibbs Sampler with Bayesian Repulsion.
    
    Parameters:
        X : ndarray (N x p)
            Data points.
        K : int
            Number of clusters.
        num_iterations : int
            Total iterations.
        h : function
            Repulsive prior function.
        burn_in : int
            Number of initial samples to discard.
        bmi_bounds : ndarray
            Array of BMI boundaries (e.g., [-1.57149027, -0.71523168, -0.1036184, inf]).
        sbp_bounds : ndarray
            Array of SBP boundaries (e.g., [0.95012032, 2.12990549, inf]).
    
    Returns:
        samples : list
            List of samples after burn-in.
    """
    N, p = X.shape  # p should be 2 for BMI and SBP

    # --- Initialization ---
    pi = dirichlet.rvs(np.ones(K) * 2, size=1).flatten()  # Randomized mixing proportions
    mu = np.random.multivariate_normal(np.mean(X, axis=0), np.cov(X.T), size=K)  # Randomized means
    Sigma = np.array([np.cov(X.T) for _ in range(K)]) 

    # Hyperparameters for priors
    alpha = np.ones(K) * 5  # Dirichlet prior for mixing proportions
    m0 = np.mean(X, axis=0)  # Prior mean for mu
    V0 = 5 * np.eye(p)       # Prior covariance for mu
    nu0 = p + 2              # Degrees of freedom for Inv-Wishart
    S0 = np.cov(X.T) * 2     # Scale matrix for Inv-Wishart

    samples = []  # List to store sampled parameters

    # --- Gibbs Sampling Loop ---
    for iteration in tqdm(range(num_iterations), desc="Sampling"):
        # Step 1: Sample z_i (Cluster Assignments)
        log_posterior = np.log(np.clip(pi, 1e-10, None)) + np.array(
            [multivariate_normal.logpdf(X, mean=mu[k], cov=Sigma[k]) for k in range(K)]
        ).T
        posterior_probs = np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True))
        posterior_probs /= posterior_probs.sum(axis=1, keepdims=True)
        z = np.array([np.random.choice(K, p=p) for p in posterior_probs])

        # --- Step 2: Update mu_k (Cluster Means) with neighbor block proposal including current block ---
        for k in range(K):
            X_k = X[np.array(z) == k]  # Data points in cluster k
            n_k = len(X_k)
            if n_k > 0:
                # Determine current bin for each dimension
                current_bin_bmi = np.digitize(mu[k, 0], bins=bmi_bounds)
                current_bin_sbp = np.digitize(mu[k, 1], bins=sbp_bounds)
                
                # Candidate bins for forward proposal (include current bin)
                candidate_bins_bmi = [i for i in [current_bin_bmi - 1, current_bin_bmi, current_bin_bmi + 1]
                                    if 1 <= i <= len(bmi_bounds) - 1]
                candidate_bins_sbp = [i for i in [current_bin_sbp - 1, current_bin_sbp, current_bin_sbp + 1]
                                    if 1 <= i <= len(sbp_bounds) - 1]
                
                # Number of candidate bins in each dimension (for forward proposal)
                num_cand_bmi = len(candidate_bins_bmi)
                num_cand_sbp = len(candidate_bins_sbp)
                
                # Randomly choose a candidate bin for each dimension (forward)
                new_bin_bmi = np.random.choice(candidate_bins_bmi)
                new_bin_sbp = np.random.choice(candidate_bins_sbp)
                
                # Determine boundaries for the chosen forward candidate bins
                lower_bmi = bmi_bounds[new_bin_bmi - 1]
                upper_bmi = bmi_bounds[new_bin_bmi]
                lower_sbp = sbp_bounds[new_bin_sbp - 1]
                upper_sbp = sbp_bounds[new_bin_sbp]
                
                # Replace infinite bounds with fixed margins (if necessary)
                if np.isinf(lower_bmi): lower_bmi = -2
                if np.isinf(upper_bmi): upper_bmi = 6
                if np.isinf(lower_sbp): lower_sbp = -3
                if np.isinf(upper_sbp): upper_sbp = 5
                
                # Forward proposal density: uniform over the chosen block,
                # divided by the total number of candidate blocks in each dimension.
                q_old = 1.0 / (num_cand_bmi * num_cand_sbp)
                
                # Sample uniformly within the chosen forward candidate block:
                mu_proposed = np.array([
                    np.random.uniform(lower_bmi, upper_bmi),
                    np.random.uniform(lower_sbp, upper_sbp)
                ])
                
                # Now compute the reverse proposal density: q(mu_old | mu_proposed)
                proposed_bin_bmi = np.digitize(mu_proposed[0], bins=bmi_bounds)
                proposed_bin_sbp = np.digitize(mu_proposed[1], bins=sbp_bounds)
                candidate_bins_bmi_new = [i for i in [proposed_bin_bmi - 1, proposed_bin_bmi, proposed_bin_bmi + 1]
                                        if 1 <= i <= len(bmi_bounds) - 1]
                candidate_bins_sbp_new = [i for i in [proposed_bin_sbp - 1, proposed_bin_sbp, proposed_bin_sbp + 1]
                                        if 1 <= i <= len(sbp_bounds) - 1]
                num_cand_bmi_new = len(candidate_bins_bmi_new)
                num_cand_sbp_new = len(candidate_bins_sbp_new)
              
                q_new = 1.0 / (num_cand_bmi_new * num_cand_sbp_new)
                
                # Compute the target densities (log domain) at the proposed and current states.
                proposed_mu = mu.copy()
                proposed_mu[k] = mu_proposed
                log_target_proposed = (
                    multivariate_normal.logpdf(mu_proposed, mean=m0, cov=V0)
                    + np.log(h(proposed_mu))
                    + np.sum(multivariate_normal.logpdf(X_k, mean=mu_proposed, cov=Sigma[k]))
                )
                log_target_current = (
                    multivariate_normal.logpdf(mu[k], mean=m0, cov=V0)
                    + np.log(h(mu))
                    + np.sum(multivariate_normal.logpdf(X_k, mean=mu[k], cov=Sigma[k]))
                    + 1e-6
                )
                
                # Compute the acceptance ratio with the proposal densities included:
                log_acceptance_rate = (log_target_proposed + np.log(q_old)) - (log_target_current + np.log(q_new))
                acceptance_rate = np.exp(np.clip(log_acceptance_rate, -100, 0))
                if np.random.rand() < min(1, acceptance_rate):
                    mu[k] = mu_proposed
            else:
                mu[k] = np.random.multivariate_normal(m0, V0)

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



