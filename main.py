from scipy.stats import norm, truncnorm
import numpy as np
from tqdm import tqdm


def get_dists(n):
    dists = []
    for _ in range(n):
        mean = np.random.rand() * 100
        std = np.random.rand() * 10
        complete_dist = norm(mean, std)
        interval = sorted([complete_dist.rvs(), complete_dist.rvs()])
        truncated_dist = truncnorm(interval[0], interval[1], loc=mean, scale=std)
        dists.append(truncated_dist)
    return dists


def get_x_i(dists):
    x_i = []
    for d in dists:
        x_i.extend(d.rvs(1))
    return x_i


def prophet_lambda(x_i, dists):
    # find the mean of the distribution of max(x_i) as lambda
    argmax_x_i = np.argmax(x_i)
    lam = (1 / 2) * dists[argmax_x_i].mean()
    for x in x_i:
        if x >= lam:
            return x


def prophet_eta(x_i, dists):
    # find the median of the distribution of max(x_i) as eta
    argmax_x_i = np.argmax(x_i)
    eta = dists[argmax_x_i].ppf(0.5)
    for x in x_i:
        if x >= eta:
            return x


if __name__ == "__main__":
    n = 20
    k = 1
    num_samples = 10000
    dists = get_dists(n)

    results_lambda = []
    results_eta = []
    results_prophet = []

    for _ in tqdm(range(num_samples)):
        x_i = get_x_i(dists)
        lam_result = prophet_lambda(x_i, dists)
        eta_result = prophet_eta(x_i, dists)
        if lam_result is not None:
            results_lambda.append(lam_result)
        if eta_result is not None:
            results_eta.append(eta_result)
        results_prophet.append(np.max(x_i))

    print(len(results_lambda))
    print(len(results_eta))
    print(np.mean(results_lambda) / np.mean(results_prophet))
    print(np.mean(results_eta) / np.mean(results_prophet))
