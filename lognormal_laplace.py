from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
from tqdm import tqdm

from estimators import laplace_mech

"""
The purpose of this script is to produce Figure 1. This experiment
shows the tradeoff between bias and error at different privacy levels
on a synthetic dataset.
"""

# Fixed seed for reproducibility
np.random.seed(0)

# Produce synthetic log-normal dataset of size N.
def sample(mu, sigma, N):
	return np.exp(np.random.normal(mu, sigma, N))

# PDF of above sampling distribution.
def pdf(x, mu, sigma):
	return 1 / (sigma * x * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((np.log(x) - mu) / sigma) ** 2)

# check if results have already been computed
load_name = input("load results: ")

if load_name:
	npz = np.load(load_name + ".npz")
	D = npz["D"]           # recover (mean-shifted) estimates
	T = npz["T"]           # recover clipping thresholds
	eps = npz["eps"]       # recover eps settings
	median = npz["median"] # recover median used for synthetic sample
	sigma = npz["sigma"]   # recover sig parameter for synthetic sample
else:
	save_name = input("save results: ")

	# === EXPERIMENT PARAMS ===
	N = 500                            # sample size
	M = 10000                          # no repetitions to estimate properties of estimator
	median = 60000                     # median of lognormal population
	sigma = 1                          # population drawn from log of normal with std sigma
	eps = np.array([0.01, 0.05, 0.1])  # privacy params (plot code assumes 3 values)
	T = np.linspace(20000, 600000, 50) # range of truncation thresholds
	# =========================

	# repeatedly run estimator; zero out mean since we only care about bias
	mean = np.exp(np.log(median) + sigma ** 2 / 2)
	D = np.array([
		[[laplace_mech(sample(np.log(median), sigma, N), t, e) for _ in range(M)] for t in tqdm(T)]

		for e in tqdm(eps)
	]) - mean

	# save results to avoid recomputing when adjusting plots
	np.savez(save_name + ".npz", D = D, T = T, eps = eps, median = median, sigma = sigma)

B = np.mean(D, -1)               # estimated bias (for each val of epsilon)
S = np.sqrt(np.var(D, -1))       # estimated standard err (for each val of epsilon)
R = np.sqrt(np.mean(D ** 2, -1)) # estimated RMSE (for each val of epsilon)

rescale = 1 # rescale clipping threshold
H = 50000   # number of points for showing population PDF in background

fig, axs = plt.subplots(1, 3, figsize = (12, 4))
for i in range(len(eps)):
	ax = axs[i]

	x = np.linspace(1 / H, np.max(T), H)
	y = pdf(x, np.log(median), sigma)
	pdf_ax = ax.twinx()
	pdf_ax.fill_between(x / rescale, y, 0, alpha = 0.1, facecolor = "r")
	pdf_ax.set_yticklabels([])
	pdf_ax.set_yticks([])

	ax.set_title(r"$\epsilon = " + str(eps[i]) + r"$")
	ax.ticklabel_format(axis = "x", style = "sci", scilimits = (0, 0), useMathText = True)
	# NOTE: for this estimator, bias is a function only of
	# the truncation threshold, not of the magnitude of noise,
	# which is in turn determined by epsilon. Moreover, since the
	# noise scales with 1/epsilon, using the bias estimate corresponding
	# to the largest value of epsilon (i.e. B[-1]) will lead to the most
	# accurate estimate for all privacy levels.
	ax.plot(T / rescale, -B[-1], linestyle = "dotted", label = "Bias")
	ax.plot(T / rescale, S[i], linestyle = "dashed", label = "Standard Error")
	ax.plot(T / rescale, R[i], linestyle = "solid", label = "RMSE")
	ax.set_xlabel(f"Clipping Threshold")
	ax.set_ylim(top = 88000, bottom = -4000)
	ax.legend(loc = "upper right")
plt.tight_layout()
plt.savefig((load_name or save_name) + ".pdf", bbox_inches = "tight")
plt.show()
