from itertools import product
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import mode
from tqdm import tqdm

from estimators import fine

"""
The purpose of this script is to help produce Figure 3. This experiment
compares our unbiased estimator to the biased Karwa-Vadhan estimator
on synthetic normal data.
"""

# Fixed seed for reproducibility
np.random.seed(0)

# Produce synthetic normal dataset with mean mu, sigma 1, and size N.
def sample(mu, N):
	return np.random.normal(mu, 1, N)

# Check if results have already been computed
load_name = input("load results: ")

if load_name:
	data = np.load(load_name + ".npz")
	mus = data["mus"]                 # recover range of population means
	unbiased_mu = data["unbiased_mu"] # recover unbiased estimates of mus
	biased_mu = data["biased_mu"]     # recover biased estimates of mus
else:
	save_name = input("save results: ")

	# === EXPERIMENT PARAMS ===
	mus = np.linspace(0, 2, 41) # range of true means
	M = 1000000                 # no repetitions to estimate properties of estimator
	N = 400                     # sample size
	eps = 1                     # privacy param
	delta = 1 / N / 10          # privacy failure rate (should be o(1/N))
	c = 1                       # clipping radius around rough estimate
	n1 = N // 4                 # no samples for coarse estimation (shouldn't waste too many on this step)
	# =========================

	# repeatedly run both estimators
	unbiased_mu = np.array([
		np.array([fine(sample(mu, N), eps, delta, 1, c, n1, unbiased = True) for _ in range(M)])
		for mu in tqdm(mus)
	])
	biased_mu = np.array([
		np.array([fine(sample(mu, N), eps, delta, 1, c, n1, unbiased = False) for _ in range(M)])
		for mu in tqdm(mus)
	])

	# save results to avoid recomputing when adjusting plots
	np.savez(save_name + ".npz",
		M = M, N = N, eps = eps, delta = delta, c = c, n1 = n1,
		mus = mus, unbiased_mu = unbiased_mu, biased_mu = biased_mu)

print(np.mean(unbiased_mu, -1))
print(np.mean(biased_mu, -1))

plt.figure(figsize = (6, 4))

biased_err = 2 * np.std(biased_mu, -1) / np.sqrt(biased_mu.shape[-1])
biased_bias = np.mean(biased_mu, -1)
plt.plot(mus, biased_bias - mus, label = "Karwa-Vadhan")
plt.fill_between(mus, biased_bias - mus - biased_err, biased_bias - mus + biased_err, alpha = 0.5)

unbiased_err = 2 * np.std(unbiased_mu, -1) / np.sqrt(unbiased_mu.shape[-1])
unbiased_bias = np.mean(unbiased_mu, -1)
plt.plot(mus, unbiased_bias - mus, label = "Bias-Corrected")
plt.fill_between(mus, unbiased_bias - mus - unbiased_err, unbiased_bias - mus + unbiased_err, alpha = 0.5)

plt.xlabel(f"$\\mu$")
plt.ylabel("Signed Bias")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.savefig((load_name or save_name) + ".pdf", bbox_inches = "tight")
plt.show()
