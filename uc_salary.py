import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# Fixed seed for reproducibility
np.random.seed(0)

def laplace_mech(D, t, eps):
	return np.mean(D.clip(0, t)) + np.random.laplace(0, t / (eps * len(D)))

def subsample(A, n):
	N = A.shape[0]
	return A[np.random.choice(N, n, replace = False)]

df = pd.read_csv(input("salary dataset path (include .csv): "))
A = df["Total Pay"].to_numpy()

load_name = input("load results: ")

if load_name:
	data = np.load(load_name + ".npz")
	eps = data["eps"]
	T = data["T"]
	D = data["D"]
else:
	save_name = input("save results: ")

	# === EXPERIMENT PARAMS ===
	eps = np.array([0.01, 0.05, 0.1, 1, 2, 4]) # privacy params
	T = np.linspace(20000, 400000, 50)         # range of truncation thresholds
	M = 5000                                   # no repetitions to estimate properties of estimator
	N = 500                                    # subsampling size
	# =========================

	# repeatedly run estimator
	D = np.array([
		[[laplace_mech(subsample(A, N), t, e) - np.mean(A) for _ in range(M)] for t in tqdm(T)]

		for e in tqdm(eps)
	])

	# save results to avoid recomputing when updating plots
	np.savez(save_name + ".npz", eps = eps, T = T, D = D)

B = np.mean(D, -1)
S = np.sqrt(np.var(D, -1))
R = np.sqrt(np.mean(D ** 2, -1))

I = np.argmin(R, -1)
print(f"eps = {eps}")
print(f"T = {T[I]}")
print(f"B = {B[np.arange(len(eps)), I]}")
print(f"S = {S[np.arange(len(eps)), I]}")
print(f"R = {R[np.arange(len(eps)), I]}")

J = [1, 3] # which values of epsilon to plot (plot code assumes 2 vals)
rescale = 1
fig, axs = plt.subplots(1, 2, figsize = (8, 4))
for i, j in enumerate(J):
	ax = axs[i]

	pdf_ax = ax.twinx()
	pdf_ax.hist(A / rescale, bins = 1000, range = (0, 200000 / rescale), density = True, color = "r", alpha = 0.2)
	pdf_ax.set_yticklabels([])
	pdf_ax.set_yticks([])

	ax.set_title(r"$\epsilon = " + str(eps[j]) + r"$")
	# NOTE: for this estimator, bias is a function only of
	# the truncation threshold, not of the magnitude of noise,
	# which is in turn determined by epsilon. Moreover, since the
	# noise scales with 1/epsilon, using the bias estimate corresponding
	# to the largest value of epsilon (i.e. B[-1]) will lead to the most
	# accurate estimate for all privacy levels.
	ax.plot(T / rescale, -B[-1], linestyle = "dotted", label = "Bias")
	ax.plot(T / rescale, S[j], linestyle = "dashed", label = "Standard Error")
	ax.plot(T / rescale, R[j], linestyle = "solid", label = "RMSE")
	ax.set_xlabel(f"Clipping Threshold")
	ax.legend(loc = "upper right")
	ax.ticklabel_format(axis = "both", style = "sci", scilimits = (0, 0), useMathText = True)
plt.tight_layout()
plt.savefig((load_name or save_name) + ".pdf", bbox_inches = "tight")
plt.show()
