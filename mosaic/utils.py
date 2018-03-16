from scipy.optimize import minimize
from scipy import stats
import george
from george.kernels import ExpSquaredKernel
import numpy as np

def create_gp_model(xp):
	kernel = ExpSquaredKernel(1)
	gp = george.GP(kernel)
	gp.compute(xp)
	return gp

def expected_improvement(points, gp, samples, bigger_better=False):
	# are we trying to maximise a score or minimise an error?
	if bigger_better:
		best_sample = samples[np.argmax(samples)]

		mu, cov = gp.predict(samples, points)
		sigma = np.sqrt(cov.diagonal())

		Z = (mu-best_sample)/sigma

		ei = ((mu-best_sample) * stats.norm.cdf(Z) + sigma*stats.norm.pdf(Z))

		# want to use this as objective function in a minimiser so multiply by -1
		return -ei

	else:
		best_sample = samples[np.argmin(samples)]

		mu, cov = gp.predict(samples, points)
		sigma = np.sqrt(cov.diagonal())

		Z = (best_sample-mu)/sigma

		ei = ((best_sample-mu) * stats.norm.cdf(Z) + sigma*stats.norm.pdf(Z))

		# want to use this as objective function in a minimiser so multiply by -1
		return -ei

def next_sample(gp, samples, bounds=(0,1), bigger_better=False):
	"""Find point with largest expected improvement"""
	best_x = None
	best_ei = 0
	# EI is zero at most values -> often get trapped
	# in a local maximum -> multistarting to increase
	# our chances to find the global maximum
	for rand_x in np.random.uniform(bounds[0], bounds[1], size=30):
		res = minimize(expected_improvement, rand_x,
					   bounds=[bounds],
					   method='L-BFGS-B',
					   args=(gp, samples, bigger_better))
		if res.fun < best_ei:
			best_ei = res.fun
			best_x = res.x[0]

	return best_x
