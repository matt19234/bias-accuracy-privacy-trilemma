from itertools import product
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mode, skew
from tqdm import tqdm

from estimators import fine

"""
The purpose of this script is to help produce Figure 3. This experiment
compares our unbiased estimator to the biased Karwa-Vadhan estimator
on a real dataset (weights_heights.csv).
"""

# Fixed seed for reproducibility
np.random.seed(0)

# Subsample population without replacement to produce working
# datasets.
def subsample(A, N):
	return A[np.random.choice(A.shape[0], N, replace = False)]

# Locate dataset (we will load it later if we need it)
dataset_path = input("height dataset path (include .csv): ")

# Check if results have already been computed
load_name = input("load results: ")

if load_name:
	data = np.load(load_name + ".npz")
	mu = data["mu"]                   # recover population mean
	unbiased_mu = data["unbiased_mu"] # recover unbiased estimates of mu
	biased_mu = data["biased_mu"]     # recover biased estimates of mu
else:
	save_name = input("save results: ")

	# load dataset
	df = pd.read_csv(dataset_path)
	A = df["Height (inches)"].to_numpy() # extract height column (no further processing)
	mu = np.mean(A)
	sig = np.std(A)
	print("skewness", skew(A))

	# === EXPERIMENT PARAMS ===
	M = 1000000        # no repetitions to estimate properties of estimators
	N = 400            # subsample size
	eps = 1            # privacy param
	delta = 1 / N / 10 # privacy failure rate (should be o(1/N))
	c = 1              # clipping radius around coarse estimate (expressed here as a multiple of sig)
	n1 = N // 4        # sample size for coarse estimation (shouldn't waste too many on this step)
	# =========================

	# repeatedly run both estimators
	unbiased_mu = np.array([
		fine(subsample(A, N), eps, delta, sig, c * sig, n1, unbiased = True)

		for _ in tqdm(range(M))
	])
	biased_mu = np.array([
		fine(subsample(A, N), eps, delta, sig, c * sig, n1, unbiased = False)

		for _ in tqdm(range(M))
	])

	# save results to avoid recomputing when adjusting plots
	np.savez(save_name + ".npz",
		M = M, N = N, eps = eps, delta = delta, mu = mu, sig = sig, c = c, n1 = n1,
		unbiased_mu = unbiased_mu, biased_mu = biased_mu)

print(mu)
print(f"unb: {np.mean(unbiased_mu) - mu} +- {2 * np.std(unbiased_mu) / np.sqrt(len(unbiased_mu))} (95% CI)")
print(f"kv: {np.mean(biased_mu) - mu} +- {2 * np.std(biased_mu) / np.sqrt(len(biased_mu))} (95% CI)")

plt.figure(figsize = (6, 4))
plt.axvline(x = mu, color = "black", alpha = 0.8, linestyle = "dashed")
plt.hist(biased_mu, density = True, bins = np.linspace(67, 69.25, 300), histtype='stepfilled', alpha = 1, label = "Karwa-Vadhan")
plt.hist(unbiased_mu, density = True, bins = np.linspace(67, 69.25, 300), histtype='stepfilled', alpha = 0.9, label = "Bias-Corrected")
plt.xlabel("Height (inches)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig((load_name or save_name) + ".pdf", bbox_inches = "tight")
plt.show()
