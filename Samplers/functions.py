import numpy as np
from scipy.spatial.distance import pdist

def mu_distance(mu_samples, mode='min'):
    trace = []
    for mu in mu_samples:
        dists = pdist(mu)  # All pairwise Euclidean distances between cluster means
        if mode == 'min':
            trace.append(np.min(dists))
        elif mode == 'mean':
            trace.append(np.mean(dists))
        elif mode == 'max':
            trace.append(np.max(dists))
        else:
            raise ValueError("mode must be 'min', 'mean', or 'max'")
    return np.mean(trace)

def mu_distance_trace(mu_samples, mode='min'):
    trace = []
    for mu in mu_samples:
        dists = pdist(mu)  # All pairwise Euclidean distances between cluster means
        if mode == 'min':
            trace.append(np.min(dists))
        elif mode == 'mean':
            trace.append(np.mean(dists))
        elif mode == 'max':
            trace.append(np.max(dists))
        else:
            raise ValueError("mode must be 'min', 'mean', or 'max'")
    return np.trace(trace)