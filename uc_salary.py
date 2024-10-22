import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from estimators import laplace_mech

"""
The purpose of this script is to produce Figure 2 and Table 1. This
experiment shows the tradeoff between bias and error at different
privacy levels on a real dataset (uc_salaries_2011.csv).
"""

# Fixed seed for reproducibility
np.random.seed(0)

# Subsample population without replacement to produce working
# datasets.
def subsample(A, N):
	return A[np.random.choice(A.shape[0], N, replace = False)]

# load dataset
df = pd.read_csv(input("salary dataset path (include .csv): "))
A = df["Total Pay"].to_numpy() # extract salary column (no further processing)

# check if results have already been computed
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

	# save results to avoid recomputing when adjusting plots
	np.savez(save_name + ".npz", eps = eps, T = T, D = D)

B = np.mean(D, -1)               # estimated bias (for each truncation threshold)
S = np.sqrt(np.var(D, -1))       # estimated standard err (for each truncation threshold)
R = np.sqrt(np.mean(D ** 2, -1)) # estimated RMSE (for each truncation threshold)

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
