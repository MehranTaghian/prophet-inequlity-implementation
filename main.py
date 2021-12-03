from scipy.stats import norm, truncnorm
import numpy as np
from tqdm import tqdm
import heapq


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


def prophet(x_i, k):
    if k > 1:
        heapq.heapify(x_i)
        results = heapq.nlargest(k, x_i)
        # print("prophet:", results)
        return np.sum(results)
    else:
        return np.max(x_i)


def experiment_single_k():
    results_lambda = []
    results_eta = []
    results_prophet = []
    for _ in tqdm(range(num_samples)):
        x_i = get_x_i(dists)
        prophet_result = prophet(x_i, k)
        lam_result = prophet_lambda(x_i, dists)
        eta_result = prophet_eta(x_i, dists)
        if lam_result is not None:
            results_lambda.append(lam_result)
        if eta_result is not None:
            results_eta.append(eta_result)
        results_prophet.append(prophet_result)
    print(len(results_lambda))
    print(len(results_eta))
    print(np.mean(results_lambda) / np.mean(results_prophet))
    print(np.mean(results_eta) / np.mean(results_prophet))


def sum_prob(dists, m):
    # \sum Pr(x_i > m)
    pr = [(1 - d.cdf(m)) for d in dists]
    return np.sum(pr)


def find_threshold(dists, lr, k, initial_m=None):
    upperbound = k - np.sqrt(2 * k * np.log(k))
    if initial_m is None:
        initial_m = dists[0].rvs(1)[0]

    m_lower = m_higher = initial_m
    while True:
        # print('sum_prob_m', sum_prob_m)
        # print('upperbound', upperbound)
        # print('m', m)
        sum_prob_m_lower = sum_prob(dists, m_lower)
        sum_prob_m_higher = sum_prob(dists, m_higher)

        if sum_prob_m_lower <= upperbound:
            return m_lower
        elif sum_prob_m_higher <= upperbound:
            return m_higher

        m_lower -= lr
        m_higher += lr


def multiple_k(x_i, k, m):
    results = []
    counter_k = k
    for i in range(len(x_i)):
        if x_i[i] >= m:
            results.append(x_i[i])
            counter_k -= 1
        if len(results) == k:
            break
        elif len(x_i) - i <= counter_k:
            results.extend(x_i[i:])
            break

    # print("algo: ", results)
    return np.sum(results)


def experiment_multiple_k():
    results_algo = []
    results_prophet = []
    max_mean = dists[0].mean()
    for d in dists[1:]:
        if d.mean() > max_mean:
            max_mean = d.mean()
    m = find_threshold(dists, lr, k, max_mean)
    for _ in tqdm(range(num_samples)):
        x_i = get_x_i(dists)
        prophet_result = prophet(x_i, k)
        algo_result = multiple_k(x_i, k, m)
        results_prophet.append(prophet_result)
        if algo_result is not None:
            results_algo.append(algo_result)

    print("Upper bound beta:", 1 + np.sqrt(8 * np.log(k) / k))
    print("Lower bound beta:", 1 + np.sqrt(1 / (512 * k)))
    print("beta:", np.mean(results_prophet) / np.mean(results_algo))
    print(np.mean(results_algo) / np.mean(results_prophet))


if __name__ == "__main__":
    n = 64
    k = 8
    lr = 0.001
    num_samples = 2000
    dists = get_dists(n)

    # experiment_single_k()
    experiment_multiple_k()
