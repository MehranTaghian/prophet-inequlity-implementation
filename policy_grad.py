import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *


def plot_performance(exp_seeds, k):
    avg_alpha = np.mean(exp_seeds, axis=0)
    std_alpha = np.std(exp_seeds, axis=0) / np.sqrt(exp_seeds.shape[0])
    plt.figure(figsize=[10, 8])
    plt.plot(avg_alpha)
    plt.fill_between(avg_alpha - 2.26 * std_alpha, avg_alpha + 2.26 * std_alpha, alpha=0.2)
    plt.xlabel('Number of samples')
    plt.ylabel('Competitive ratio (Alpha)')
    plt.title(f"Performance of the bandit policy gradient method compared to the prophet for k = {k}")
    plt.savefig(f'bandit_pg_{k}.jpg', dpi=300)
    plt.show()


def run_experiments(seed, k):
    # Solution parameters
    torch.manual_seed(seed)
    np.random.seed(seed)

    r = get_dists(na, mean_interval=mean_interval, std_interval=std_interval)
    pref = torch.randn(na, requires_grad=True)
    opt = torch.optim.SGD([pref], lr=0.1)

    prefs = np.zeros((T, na))
    for t in tqdm(range(T)):
        # Interaction
        pol = torch.distributions.Categorical(logits=pref)
        # print(pref)
        # policy epsilon greedy:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * t / EPS_DECAY)
        if np.random.rand() < eps_threshold:
            A = torch.randint(na, [1])[0]
        else:
            A = pol.sample()
        R = torch.tensor(r[A].rvs())
        # Form loss
        sur_obj = pol.log_prob(A) * R
        loss = -sur_obj

        # Update
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Log
        prefs[t] = pref.data.clone()

    # Plot
    # p = plt.plot(prefs)
    # plt.xlabel('Trial')
    # plt.ylabel('Preference')
    # plt.legend(iter(p), range(na))
    # plt.show()

    # for d in range(len(r)):
    #     print(d, r[d].mean(), pref[d])

    # print("Testing ...")
    prophet_results = []
    pg_results = []
    pref = np.argsort(pref.tolist())[-k:]
    for _ in tqdm(range(num_samples)):
        x_i = get_x_i(r)
        prophet_results.append(prophet(x_i, k))
        pg_sum = 0
        for i in reversed(pref):
            pg_sum += x_i[i]
        pg_results.append(pg_sum)

    print(np.mean(pg_results) / np.mean(prophet_results))

    return np.array(pg_results) / np.arange(1, len(pg_results) + 1), \
           np.array(prophet_results) / np.arange(1, len(prophet_results) + 1)


if __name__ == "__main__":
    num_seeds = 10
    na = 64
    mean_interval = 10
    std_interval = 1
    ks = [1, 2, 8, 16]

    num_samples = 1000

    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200

    T = 50000

    pg_seeds, prophet_seeds = [], []

    for k in ks:
        for s in range(num_seeds):
            pg_seed, prophet_seed = run_experiments(s, k)
            pg_seeds.append(pg_seed)
            prophet_seeds.append(prophet_seed)

        plot_performance(np.array(pg_seeds) / np.array(prophet_seeds), k)
