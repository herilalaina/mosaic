"""Utils file."""

import george
import numpy as np

from george.kernels import ExpSquaredKernel
from env import Env
from space import Space, Node_space
from scipy.optimize import minimize
from scipy import stats


def create_custom_space_env(info_space, info_env):
    """Function to create custom space and envrionment."""
    custom_space = Space()
    custom_env = Env
    if len(info_space) != len(info_env):
        raise Exception("Info space and info env don't have the same length.")

    parent_node = Node_space(info_space[0])
    custom_space.root.append_child(parent_node)
    for node_name in info_space[1:]:
        child_node = parent_node.add_child(node_name)
        parent_node = child_node

    custom_space.terminal_pointer = [parent_node]

    

def create_gp_model(xp):
    """Create gaussian process."""
    kernel = ExpSquaredKernel(1)
    gp = george.GP(kernel)
    gp.compute(xp)
    return gp


def expected_improvement(points, gp, samples, bigger_better=False):
    """Are we trying to maximise a score or minimise an error?."""
    if bigger_better:
        best_sample = samples[np.argmax(samples)]

        mu, cov = gp.predict(samples, points)
        sigma = np.sqrt(cov.diagonal())

        Z = (mu-best_sample)/sigma

        ei = ((mu-best_sample) * stats.norm.cdf(Z) + sigma*stats.norm.pdf(Z))

        # want to use this as objective function in a minimiser
        # so multiply by -1
        return -ei

    else:
        best_sample = samples[np.argmin(samples)]

        mu, cov = gp.predict(samples, points)
        sigma = np.sqrt(cov.diagonal())

        Z = (best_sample-mu)/sigma

        ei = ((best_sample-mu) * stats.norm.cdf(Z) + sigma*stats.norm.pdf(Z))

        # want to use this as objective function in a minimiser
        # so multiply by -1
        return -ei


def next_sample(gp, samples, bounds=(0, 1), bigger_better=False):
    """Find point with largest expected improvement."""
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
