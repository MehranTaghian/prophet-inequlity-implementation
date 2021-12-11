import numpy as np
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt


def plot_performance(exp_seeds, k):
    exp_seeds = np.array(exp_seeds)
    avg_alpha = np.mean(exp_seeds, axis=0)
    std_alpha = np.std(exp_seeds, axis=0) / np.sqrt(exp_seeds.shape[0])

    plt.figure(figsize=[10, 8])
    plt.plot(avg_alpha)
    plt.fill_between(avg_alpha - 2.26 * std_alpha, avg_alpha + 2.26 * std_alpha, alpha=0.2)
    plt.xlabel('Number of samples')
    plt.ylabel('Competitive ratio (Alpha)')
    plt.title("Performance of the threshold algorithm when k > 1")
    plt.savefig(f'k-{k}-search.jpg', dpi=300)
    plt.show()


def plot_beta(exp_seeds, k):
    upperbound_beta = 1 + np.sqrt(8 * np.log(k) / k)
    lowerbound_beta = 1 + np.sqrt(1 / (512 * k))
    exp_seeds = 1 / np.array(exp_seeds)
    avg_beta = np.mean(exp_seeds, axis=0)
    std_beta = np.std(exp_seeds, axis=0) / np.sqrt(exp_seeds.shape[0])

    plt.figure(figsize=[10, 8])
    plt.plot(avg_beta, label='beta')
    plt.fill_between(avg_beta - 2.26 * std_beta, avg_beta + 2.26 * std_beta, alpha=0.2)
    plt.hlines(upperbound_beta, xmin=0, xmax=exp_seeds.shape[1], linestyles='dashed', label='Upperbound beta',
               colors='red')
    plt.hlines(lowerbound_beta, xmin=0, xmax=exp_seeds.shape[1], linestyles='dashed', label='Lowerbound beta',
               colors='orange')
    plt.xlabel('Number of samples')
    plt.ylabel(f'Beta value given k = {k}')
    plt.title("Beta value and the valid range for beta")
    plt.legend(loc='lower right')
    plt.savefig(f'beta-{k}.jpg', dpi=300)
    plt.show()


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


def experiment_multiple_k(k, seeds=10):
    seed_results = []
    for s in range(seeds):
        dists = get_dists(n, mean_interval=mean_interval, std_interval=std_interval)
        np.random.seed(s)
        results_algo = []
        results_prophet = []
        seed_result = []
        max_mean = dists[0].mean()
        for d in dists[1:]:
            if d.mean() > max_mean:
                max_mean = d.mean()
        m = find_threshold(dists, lr, k, max_mean)
        for _ in tqdm(range(num_samples)):
            x_i = get_x_i(dists)
            results_prophet.append(prophet(x_i, k))
            results_algo.append(multiple_k(x_i, k, m))
            seed_result.append(np.mean(results_algo) / np.mean(results_prophet))

        seed_results.append(seed_result)
        # print("Upper bound beta:", 1 + np.sqrt(8 * np.log(k) / k))
        # print("Lower bound beta:", 1 + np.sqrt(1 / (512 * k)))
        # print("beta:", np.mean(results_prophet) / np.mean(results_algo))
        # print(np.mean(results_algo) / np.mean(results_prophet))

    plot_performance(seed_results, k)
    plot_beta(seed_results, k)


if __name__ == "__main__":
    n = 64
    ks = [2, 8, 16]
    lr = 0.001
    seeds = 10
    num_samples = 1000
    mean_interval = 10
    std_interval = 1
    for k in ks:
        experiment_multiple_k(k, seeds)
