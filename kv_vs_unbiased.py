from itertools import product
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import mode
# from scipy.signal import savgol_filter
from tqdm import tqdm

# Fixed seed for reproducibility
np.random.seed(0)

def sample(mu, sigma, N):
	return np.random.normal(mu, sigma, N)

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

def fine_1d(D, eps, delta, c, n1, unbiased):
	mu_tilde = coarse_1d(D[:n1], eps, delta, unbiased)

	n2 = D.shape[0] - n1

	if np.isnan(mu_tilde):
		B = np.random.binomial(1, delta, n2)
		return 1/delta * np.mean(B * D[n1:])
	else:
		return np.mean(D[n1:].clip(mu_tilde - c, mu_tilde + c)) + np.random.laplace(0, 2 * c / (n2 * eps))

def fine(D, eps, delta, c, n1, unbiased):
	A = np.empty(D.shape[:-1])
	for i in np.ndindex(A.shape):
		A[i] = fine_1d(D[i], eps, delta, c, n1, unbiased)
	return A

def D(mu, N):
	return np.random.normal(mu, 1, N)

load_name = input("load results: ")

if load_name:
	data = np.load(load_name + ".npz")
	mus = data["mus"]
	unbiased_mu = data["unbiased_mu"]
	biased_mu = data["biased_mu"]
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
		np.array([fine(D(mu, N), eps, delta, c, n1, unbiased = True) for _ in range(M)])
		for mu in tqdm(mus)
	])
	biased_mu = np.array([
		np.array([fine(D(mu, N), eps, delta, c, n1, unbiased = False) for _ in range(M)])
		for mu in tqdm(mus)
	])

	# save results to avoid recomputing when updating plots
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
