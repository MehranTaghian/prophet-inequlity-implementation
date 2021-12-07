import numpy as np
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt


def plot_performance(seed_eta, seed_lambda):
    seed_lambda = np.array(seed_lambda)
    seed_eta = np.array(seed_eta)

    avg_alpha_lambda = np.mean(seed_lambda, axis=0)
    avg_alpha_eta = np.mean(seed_eta, axis=0)

    std_alpha_lambda = np.std(seed_lambda, axis=0) / np.sqrt(seed_lambda.shape[0])
    std_alpha_eta = np.std(seed_eta, axis=0) / np.sqrt(seed_eta.shape[0])

    plt.figure(figsize=[10, 8])
    plt.plot(avg_alpha_lambda, label='lambda')
    plt.fill_between(avg_alpha_lambda - 2.26 * std_alpha_lambda, avg_alpha_lambda + 2.26 * std_alpha_lambda,
                     alpha=0.2)
    plt.plot(avg_alpha_eta, label='eta')
    plt.fill_between(avg_alpha_eta - 2.26 * std_alpha_eta, avg_alpha_eta + 2.26 * std_alpha_eta,
                     alpha=0.2)
    plt.xlabel('Number of samples')
    plt.ylabel('Competitive ratio (Alpha)')
    plt.legend(loc='lower right')
    plt.title("The competitive ratio of the fixed-threshold algorithms for k = 1 in prophet inequality")
    plt.savefig('k-1-search.jpg', dpi=300)
    plt.show()


def lambda_threshold_alg(x_i, dists):
    # find the mean of the distribution of max(x_i) as lambda
    argmax_x_i = np.argmax(x_i)
    lam = (1 / 2) * dists[argmax_x_i].mean()
    for x in x_i:
        if x >= lam:
            return x


def eta_threshold_alg(x_i, dists):
    # find the median of the distribution of max(x_i) as eta
    argmax_x_i = np.argmax(x_i)
    eta = dists[argmax_x_i].ppf(0.5)
    for x in x_i:
        if x >= eta:
            return x


def experiment_single_k(seeds=10):
    seed_lambda = []
    seed_eta = []
    for s in range(seeds):
        np.random.seed(s)
        dists = get_dists(n)
        performance_lambda = []
        performance_eta = []
        results_lambda = []
        results_eta = []
        results_prophet = []
        for _ in tqdm(range(num_samples)):
            x_i = get_x_i(dists)
            prophet_result = prophet(x_i, k)
            lam_result = lambda_threshold_alg(x_i, dists)
            eta_result = eta_threshold_alg(x_i, dists)
            if lam_result is not None:
                results_lambda.append(lam_result)
            if eta_result is not None:
                results_eta.append(eta_result)
            results_prophet.append(prophet_result)
            prophet_performance = np.mean(results_prophet)
            performance_lambda.append(np.mean(results_lambda) / prophet_performance)
            performance_eta.append(np.mean(results_eta) / prophet_performance)

        seed_lambda.append(performance_lambda)
        seed_eta.append(performance_eta)

    plot_performance(seed_lambda, seed_eta)


if __name__ == "__main__":
    n = 64
    k = 1
    num_samples = 1000
    num_seeds = 10
    experiment_single_k(seeds=num_seeds)
