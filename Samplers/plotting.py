# plotting.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import multivariate_normal

def plot_scatter(last_sample, X, colors, ax):
    """
    Plots a scatter plot for the last iteration of the chain.
    
    Parameters:
        last_sample : tuple
            A tuple (pi, mu, Sigma, z) from the sampler.
        X : ndarray
            Data array (N x p).
        colors : list
            List of colors for each cluster.
        ax : matplotlib.axes.Axes
            Axis on which to plot.
    """
    _, mu, _, z = last_sample
    for k in range(len(mu)):
        ax.scatter(X[z == k, 0], X[z == k, 1],
                   color=colors[k], alpha=0.7, label=f"Cluster {k+1}")
    ax.set_xlabel("BMI")
    ax.set_ylabel("SBP")
    ax.grid()
    ax.legend()


def plot_trace(run_samples, K, ax_array, burn_in):
    """
    Plots trace plots for mixing proportions, cluster means, and variances.
    
    Parameters:
        run_samples : list
            List of samples from the sampler.
        K : int
            Number of clusters.
        ax_array : array-like (1D flattened)
            Array of matplotlib.axes.Axes objects arranged in *columns* (parameters) per run row.
        burn_in : int or None
            If provided, a vertical line is added at the burn-in iteration.
    """
    pi_samples = np.array([s[0] for s in run_samples])   # (iterations, K)
    mu_samples = np.array([s[1] for s in run_samples])    # (iterations, K, p)
    sigma_samples = np.array([s[2] for s in run_samples]) # (iterations, K, p, p)

    D = mu_samples.shape[2]  # dimension 

    # Figure out how many plots per run:
    num_plots_per_run = 1 + D + D  # (pi + mu + sigma)
    
    # Sanity check: ax_array should have num_plots_per_run elements
    assert len(ax_array) == num_plots_per_run, "Mismatch between axes and parameters to plot."
    idx = 0

    # Trace for mixing proportions (pi)
    for k in range(K):
        ax_array[idx].plot(pi_samples[:, k], label=f'Cluster {k+1}')
    if burn_in is not None:
        ax_array[idx].axvline(x=burn_in, color='black', linestyle='--')
    ax_array[idx].set_title("Mixing Proportions (pi)")
    ax_array[idx].set_xlabel("Iteration")
    ax_array[idx].set_ylabel("pi")
    ax_array[idx].legend(fontsize=6)
    ax_array[idx].grid()
    idx += 1

    # Trace for cluster means (mu)
    for dim in range(D):
        for k in range(K):
            ax_array[idx].plot(mu_samples[:, k, dim], label=f'Cluster {k+1}')
        if burn_in is not None:
            ax_array[idx].axvline(x=burn_in, color='black', linestyle='--')
        ax_array[idx].set_title(f"Means (mu), Dim {dim+1}")
        ax_array[idx].set_xlabel("Iteration")
        ax_array[idx].set_ylabel(f"mu_{dim+1}")
        ax_array[idx].legend(fontsize=6)
        ax_array[idx].grid()
        idx += 1

    # Trace for variances (diagonal elements of Sigma)
    for dim in range(D):
        for k in range(K):
            var_trace = [sigma_samples[i, k][dim, dim] for i in range(len(sigma_samples))]
            ax_array[idx].plot(var_trace, label=f'Cluster {k+1}')
        if burn_in is not None:
            ax_array[idx].axvline(x=burn_in, color='black', linestyle='--')
        ax_array[idx].set_title(f"Variances (sigma²), Dim {dim+1}")
        ax_array[idx].set_xlabel("Iteration")
        ax_array[idx].set_ylabel(f"Var_{dim+1}")
        ax_array[idx].legend(fontsize=6)
        ax_array[idx].grid()
        idx += 1


def plot_with_reference_lines(mu_samples, sigma_samples, X, bmi_bounds, sbp_bounds, ax):
    """
    Plots clusters with posterior means and confidence ellipses.
    
    Parameters:
        mu_samples : ndarray
            Array of sampled cluster means.
        sigma_samples : ndarray
            Array of sampled cluster covariances.
        X : ndarray
            Data array of shape (N,2), where column 0 is BMI and column 1 is SBP.
        bmi_bounds : list
            List of BMI boundary values.
        sbp_bounds : list
            List of SBP boundary values.
        ax : matplotlib.axes.Axes
            Axis on which to plot.
    """
    posterior_mu = np.mean(mu_samples, axis=0)
    posterior_sigma = np.mean(sigma_samples, axis=0)

    # Plot the data points
    ax.scatter(X[:, 0], X[:, 1], alpha=0.7, color='lightgray', label='Data Points')

    # Plot each cluster's posterior mean and confidence ellipse
    for k in range(len(posterior_mu)):
        ax.scatter(posterior_mu[k, 0], posterior_mu[k, 1], color='black', s=100, marker='x',
                   label=f"Cluster {k+1} Mean" if k == 0 else None)
        cov_matrix = posterior_sigma[k]
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        for scale in [0.25, 0.5, 0.75, 1.0]:
            width, height = 2 * scale * np.sqrt(eigenvalues)
            ellipse = patches.Ellipse(posterior_mu[k], width, height, angle=angle,
                                      alpha=0.5, edgecolor='black', facecolor='none')
            ax.add_patch(ellipse)
            
    # Add reference lines for BMI and SBP boundaries
    for bmi_bound in bmi_bounds:
        ax.axvline(x=bmi_bound, color='red', linestyle='--', linewidth=1,
                   label='BMI Boundaries' if bmi_bound == bmi_bounds[0] else None)
    for sbp_bound in sbp_bounds:
        ax.axhline(y=sbp_bound, color='blue', linestyle='--', linewidth=1,
                   label='SBP Boundaries' if sbp_bound == sbp_bounds[0] else None)
    
    ax.set_xlabel("BMI")
    ax.set_ylabel("SBP")
    ax.grid()
    ax.legend()



def ellipse_mean(all_run_samples, X, bmi_bounds, sbp_bounds, burn_in=0, cols=2, figsize_scale=4):
    """
    Plot posterior mean clusters and confidence ellipses for multiple runs after burn-in.
    
    Parameters:
        all_run_samples : list
            List of runs; each run is a list of samples (pi, mu, Sigma, z).
        X : ndarray
            Data array (N, 2), columns are BMI and SBP.
        bmi_bounds : list or ndarray
            List of BMI boundary values.
        sbp_bounds : list or ndarray
            List of SBP boundary values.
        burn_in : int
            Number of initial samples to discard.
        cols : int
            Number of columns for subplots layout.
        figsize_scale : float
            Scale factor for figure size.
    """
    num_runs = len(all_run_samples)
    rows = (num_runs + cols - 1) // cols  # Compute number of rows
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_scale * cols, figsize_scale * rows), sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    for run_idx, (ax, run_samples) in enumerate(zip(axes, all_run_samples)):
        run_samples_burned = run_samples[burn_in:]
        mu_samples = np.array([s[1] for s in run_samples_burned])    # shape: (T, K, 2)
        sigma_samples = np.array([s[2] for s in run_samples_burned]) # shape: (T, K, 2, 2)

        posterior_mu = np.mean(mu_samples, axis=0)       # (K, 2)
        posterior_sigma = np.mean(sigma_samples, axis=0) # (K, 2, 2)

        # --- Plot ---
        ax.scatter(X[:, 0], X[:, 1], alpha=0.7, color='lightgray', label='Data Points')

        for k in range(posterior_mu.shape[0]):
            ax.scatter(posterior_mu[k, 0], posterior_mu[k, 1], color='black', s=100, marker='x',
                       label=f"Cluster {k+1} Mean" if k == 0 else None)
            cov_matrix = posterior_sigma[k]
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            for scale in [0.25, 0.5, 0.75, 1.0]:
                width, height = 2 * scale * np.sqrt(eigenvalues)
                ellipse = patches.Ellipse(posterior_mu[k], width, height, angle=angle,
                                          alpha=0.5, edgecolor='black', facecolor='none')
                ax.add_patch(ellipse)
        
        # Reference lines for BMI and SBP bounds
        for bmi_bound in bmi_bounds:
            ax.axvline(x=bmi_bound, color='red', linestyle='--', linewidth=1,
                       label='BMI Boundaries' if bmi_bound == bmi_bounds[0] else None)
        for sbp_bound in sbp_bounds:
            ax.axhline(y=sbp_bound, color='blue', linestyle='--', linewidth=1,
                       label='SBP Boundaries' if sbp_bound == sbp_bounds[0] else None)

        ax.set_title(f"Run {run_idx + 1}", fontsize=10)
        ax.set_xlabel("BMI")
        ax.set_ylabel("SBP")
        ax.grid()
        ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle("Clusters with Confidence Ellipses (Posterior Mean)", fontsize=16, y=1.02)
    plt.show()


def ellipse_lastit(all_run_samples, X, bmi_bounds, sbp_bounds, cols=2, figsize_scale=4):
    """
    Plot clusters and confidence ellipses for the LAST iteration of each run.
    
    Parameters:
        all_run_samples : list
            List of runs; each run is a list of samples (pi, mu, Sigma, z).
        X : ndarray
            Data array (N, 2), columns are BMI and SBP.
        bmi_bounds : list or ndarray
            List of BMI boundary values.
        sbp_bounds : list or ndarray
            List of SBP boundary values.
        cols : int
            Number of columns for subplot grid.
        figsize_scale : float
            Scale factor for figure size.
    """
    num_runs = len(all_run_samples)
    rows = (num_runs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_scale * cols, figsize_scale * rows), sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    for run_idx, (ax, run_samples) in enumerate(zip(axes, all_run_samples)):
        # Extract the last sample for this run
        pi_samples = np.array([s[0] for s in run_samples])
        mu_samples = np.array([s[1] for s in run_samples])
        sigma_samples = np.array([s[2] for s in run_samples])

        last_mu = mu_samples[-1]       # (K, 2)
        last_sigma = sigma_samples[-1] # (K, 2, 2)

        # --- Plot ---
        ax.scatter(X[:, 0], X[:, 1], alpha=0.7, color='lightgray', label='Data Points')

        for k in range(last_mu.shape[0]):
            ax.scatter(last_mu[k, 0], last_mu[k, 1], color='black', s=100, marker='x',
                       label=f"Cluster {k+1} Mean" if k == 0 else None)
            cov_matrix = last_sigma[k]
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            for scale in [0.25, 0.5, 0.75, 1.0]:
                width, height = 2 * scale * np.sqrt(eigenvalues)
                ellipse = patches.Ellipse(last_mu[k], width, height, angle=angle,
                                          alpha=0.5, edgecolor='black', facecolor='none')
                ax.add_patch(ellipse)
        
        # Reference lines for BMI and SBP bounds
        for bmi_bound in bmi_bounds:
            ax.axvline(x=bmi_bound, color='red', linestyle='--', linewidth=1,
                       label='BMI Boundaries' if bmi_bound == bmi_bounds[0] else None)
        for sbp_bound in sbp_bounds:
            ax.axhline(y=sbp_bound, color='blue', linestyle='--', linewidth=1,
                       label='SBP Boundaries' if sbp_bound == sbp_bounds[0] else None)

        ax.set_title(f"Run {run_idx + 1}", fontsize=10)
        ax.set_xlabel("BMI")
        ax.set_ylabel("SBP")
        ax.grid()
        ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle("Clusters with Confidence Ellipses (Last Iteration)", fontsize=16, y=1.02)
    plt.show()

