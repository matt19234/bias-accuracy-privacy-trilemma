import numpy as np

"""
Sample mean of non-negative dataset D with points clipped to
[0, t], with enough Laplace noise added to ensure eps-privacy.
"""
def laplace_mech(D, t, eps):
	return np.mean(D.clip(0, t)) + np.random.laplace(0, t / (eps * len(D)))

"""
See Algorithm 1 (https://arxiv.org/pdf/2301.13334).

Takes a dataset D of 1d datapoints, computes a coarse
estimate of the mean of the population generating D
while satisfying (eps, delta)-DP.

If the unbiased flag is set, the estimator will be unbiased
for symmetric populations.
"""
def coarse_1d(D, eps, delta, unbiased):
	T = np.random.uniform() if unbiased else 0   # uniform random offset => unbiasedness
	X = np.round(D - T).astype(np.int64)         # group into bins
	m = np.min(X)
	Y = np.bincount(X - m)                       # histogram
	Z = Y + np.random.laplace(0, 2/eps, Y.shape) # private histogram
	Z[Y == 0] = 0                                # only privatize active bins

	I = np.argmax(Z)

	if Z[I] <= 2 + 2 * np.log(1 / delta) / eps:  # stability histogram requires thresholding
		return np.nan
	else:
		return I + m + T

"""
See Algorithm 2 (https://arxiv.org/pdf/2301.13334).

Leverages the coarse estimator to produce a consistent
(eps, delta)-DP estimate of the population mean given a
a sample D from the population.

This estimator requires prior knowledge of the population
standard deviation sig.

The estimator divides the sample D of size n1 + n2 into a
size-n1 subsample used for coarse estimation and a size-n2
subsample used for fine estimation.

The fine subsample is clipped around the coarse estimate with
radius c. The clipping radius can be tuned according to
distributional assumptions (see Theorem 4.8).

The unbiased flag determines if the coarse estimator is unbiased
for symmetric populations.
"""
def fine_1d(D, eps, delta, sig, c, n1, unbiased):
	n2 = D.shape[0] - n1

	mu_tilde = coarse_1d(D[:n1] / sig, eps, delta, unbiased) * sig

	# check for coarse estimator failure
	if np.isnan(mu_tilde):
		# "name-and-shame" estimator
		B = np.random.binomial(1, delta, n2)
		return 1/delta * np.mean(B * D[n1:])
	else:
		# clip-and-noise around coarse estimate
		return np.mean(D[n1:].clip(mu_tilde - c, mu_tilde + c)) + np.random.laplace(0, 2 * c / (n2 * eps))

"""
Given a tensor D, runs fine_1d on the last coordinate of D. This
allows us to run the estimator on many datasets simultaneously.
"""
def fine(D, eps, delta, sig, c, n1, unbiased):
	return np.apply_along_axis(lambda X: fine_1d(X, eps, delta, sig, c, n1, unbiased), -1, D)
