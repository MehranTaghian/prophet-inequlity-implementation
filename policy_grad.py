import torch as tor
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *

tor.manual_seed(0)


# TODO: normalize input distributions

na = 20
r = get_dists(na)
num_samples = 1000

# Solution parameters
pref = tor.randn(na, requires_grad=True)
opt = tor.optim.SGD([pref], lr=0.1)

# Experiment parameters
T = 10000
prefs = np.zeros((T, na))
for t in tqdm(range(T)):
    # Interaction
    pol = tor.distributions.Categorical(logits=pref)
    # print(pref)
    A = pol.sample()
    R = tor.tensor(r[A].rvs())
    # Form loss
    sur_obj = pol.log_prob(A) * R
    loss = -sur_obj

    # Update
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Log
    prefs[t] = pref.data.clone()

print("Testing ...")
prophet_results = []
pg_results = []
for _ in tqdm(range(num_samples)):
    x_i = get_x_i(r)
    prophet_results.append(np.max(x_i))
    with tor.no_grad():
        pol = tor.distributions.Categorical(logits=pref)
        pg_results.append(x_i[pol.sample()])

print(np.mean(pg_results) / np.mean(prophet_results))

# Plot
p = plt.plot(prefs)
plt.xlabel('Trial')
plt.ylabel('Preference')
plt.legend(iter(p), range(na))
plt.show()
