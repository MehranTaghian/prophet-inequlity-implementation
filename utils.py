import numpy as np
from scipy.stats import norm, truncnorm
import heapq


def get_dists(n, mean_interval=10, std_interval=1):
    dists = []
    for _ in range(n):
        mean = np.random.rand() * mean_interval
        std = np.random.rand() * std_interval
        complete_dist = norm(mean, std)
        interval = sorted([complete_dist.rvs(), complete_dist.rvs()])
        truncated_dist = truncnorm(interval[0], interval[1], loc=mean, scale=std)
        dists.append(truncated_dist)
    return dists


def get_x_i(dists):
    x_i = []
    for d in dists:
        x_i.append(d.rvs())
    return x_i


def prophet(x_i, k):
    if k > 1:
        heapq.heapify(x_i)
        results = heapq.nlargest(k, x_i)
        # print("prophet:", results)
        return np.sum(results)
    else:
        return np.max(x_i)
