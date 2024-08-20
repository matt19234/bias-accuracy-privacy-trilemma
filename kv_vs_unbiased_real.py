from itertools import product
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mode, skew
from tqdm import tqdm

# Fixed seed for reproducibility
np.random.seed(0)

def subsample(A, N):
	return A[np.random.choice(A.shape[0], N, replace = False)]

def laplace_mech(D, t, eps):
	return np.mean(D.clip(0, t)) + np.random.laplace(0, t / (eps * len(D)))

def coarse_1d(D, eps, delta, unbiased):
	T = np.random.uniform() if unbiased else 0
	X = np.round(D - T).astype(np.int64)
	m = np.min(X)
	Y = np.bincount(X - m)
	Z = Y + np.random.laplace(0, 2/eps, Y.shape)
	Z[Y == 0] = 0

	I = np.argmax(Z)

	if Z[I] <= 2 + 2 * np.log(1 / delta) / eps:
		return np.nan
	else:
		return I + m + T

def fine_1d(D, eps, delta, sig, c, n1, unbiased):
	mu_tilde = coarse_1d(D[:n1] / sig, eps, delta, unbiased) * sig

	n2 = D.shape[0] - n1

	if np.isnan(mu_tilde): # check for coarse estimator failure
		B = np.random.binomial(1, delta, n2)
		return 1/delta * np.mean(B * D[n1:]) # "name-and-shame" estimator
	else:
		return np.mean(D[n1:].clip(mu_tilde - c, mu_tilde + c)) + np.random.laplace(0, 2 * c / (n2 * eps))

def fine(D, eps, delta, sig, c, n1, unbiased):
	A = np.empty(D.shape[:-1])
	for i in np.ndindex(A.shape):
		A[i] = fine_1d(D[i], eps, delta, sig, c, n1, unbiased)
	return A

dataset_path = input("height dataset path (include .csv): ")
load_name = input("load results: ")

if load_name:
	data = np.load(load_name + ".npz")
	mu = data["mu"]
	unbiased_mu = data["unbiased_mu"]
	biased_mu = data["biased_mu"]
else:
	save_name = input("save results: ")

	# load dataset
	df = pd.read_csv(dataset_path)
	A = df["Height (inches)"].to_numpy()
	mu = np.mean(A)
	sig = np.std(A)
	# A = np.concatenate((A, 2 * np.mean(A) - A))
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

	# save results to avoid recomputing when updating plots
	np.savez(save_name + ".npz",
		M = M, N = N, eps = eps, delta = delta, mu = mu, sig = sig, c = c, n1 = n1,
		unbiased_mu = unbiased_mu, biased_mu = biased_mu)

print(mu)
print(f"unb: {np.mean(unbiased_mu) - mu} +- {2 * np.std(unbiased_mu) / np.sqrt(len(unbiased_mu))} (95% CI)")
print(f"kv: {np.mean(biased_mu) - mu} +- {2 * np.std(biased_mu) / np.sqrt(len(biased_mu))} (95% CI)")

plt.figure(figsize = (6, 4))
plt.hist(biased_mu, density = True, bins = np.linspace(67, 69.25, 300), histtype='stepfilled', alpha = 1, label = "Karwa-Vadhan")
plt.hist(unbiased_mu, density = True, bins = np.linspace(67, 69.25, 300), histtype='stepfilled', alpha = 0.9, label = "Bias-Corrected")
plt.xlabel("Height (inches)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig((load_name or save_name) + ".pdf", bbox_inches = "tight")
plt.show()
