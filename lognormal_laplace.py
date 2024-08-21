from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
from tqdm import tqdm

# Fixed seed for reproducibility
np.random.seed(0)

def sample(mu, sigma, N):
	return np.exp(np.random.normal(mu, sigma, N))

def pdf(x, mu, sigma):
	return 1 / (sigma * x * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((np.log(x) - mu) / sigma) ** 2)

def laplace_mech(D, t, eps):
	return np.mean(D.clip(0, t)) + np.random.laplace(0, t / (eps * len(D)))

load_name = input("load results: ")

if load_name:
	npz = np.load(load_name + ".npz")
	D = npz["D"]
	T = npz["T"]
	eps = npz["eps"]
	median = npz["median"]
	sigma = npz["sigma"]
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

	# save results to avoid recomputing when updating plots
	np.savez(save_name + ".npz", D = D, T = T, eps = eps, median = median, sigma = sigma)

B = np.mean(D, -1)
S = np.sqrt(np.var(D, -1))
R = np.sqrt(np.mean(D ** 2, -1))

rescale = 1
H = 50000

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
