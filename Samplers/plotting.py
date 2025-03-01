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


def plot_trace(run_samples, K, ax_array):
    """
    Plots trace plots for mixing proportions, cluster means, and variances.
    
    Parameters:
        run_samples : list
            List of samples from the sampler.
        K : int
            Number of clusters.
        ax_array : array-like
            Array of matplotlib.axes.Axes objects arranged as rows:
              - Row 0: Mixing proportions (pi)
              - Rows 1 to (D): Cluster means (each dimension, D = number of dimensions)
              - Subsequent rows: Diagonal variances from Sigma (one row per dimension)
    """
    pi_samples = np.array([s[0] for s in run_samples])
    mu_samples = np.array([s[1] for s in run_samples])
    sigma_samples = np.array([s[2] for s in run_samples])
    
    # Trace for mixing proportions (pi)
    for k in range(K):
        ax_array[0].plot(pi_samples[:, k], label=f'Cluster {k+1}')
    ax_array[0].set_title("Mixing Proportions (pi)")
    ax_array[0].set_xlabel("Iteration")
    ax_array[0].set_ylabel("pi")
    ax_array[0].legend(fontsize=8)
    ax_array[0].grid()

    # Trace for cluster means (mu): plot each dimension separately
    D = mu_samples.shape[2]
    for dim in range(D):
        for k in range(K):
            ax_array[dim+1].plot(mu_samples[:, k, dim], label=f'Cluster {k+1}')
        ax_array[dim+1].set_title(f"Cluster Means (mu), Dimension {dim+1}")
        ax_array[dim+1].set_xlabel("Iteration")
        ax_array[dim+1].set_ylabel(f"mu_{dim+1}")
        ax_array[dim+1].legend(fontsize=8)
        ax_array[dim+1].grid()
    
    # Trace for variances: for each dimension (diagonal elements of Sigma)
    for dim in range(D):
        for k in range(K):
            var_trace = [sigma_samples[i, k][dim, dim] for i in range(len(sigma_samples))]
            ax_array[dim+1+D].plot(var_trace, label=f'Cluster {k+1}')
        ax_array[dim+1+D].set_title(f"Cluster Variances, Dimension {dim+1}")
        ax_array[dim+1+D].set_xlabel("Iteration")
        ax_array[dim+1+D].set_ylabel(f"Variance_{dim+1}")
        ax_array[dim+1+D].legend(fontsize=8)
        ax_array[dim+1+D].grid()


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
    ax.scatter(X[:, 0], X[:, 1], alpha=0.7, color='grey', label='Data Points')

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