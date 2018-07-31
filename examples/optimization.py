import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import math

from scipy.optimize import basinhopping, differential_evolution
from mosaic.mosaic import Search
from mosaic.simulation.scenario import ListTask, ChoiceScenario

# Function to optimize
def Michalewicz(coeffs):
    m = 20 # the larger the number the narrower the channel
    n = range(len(coeffs))
    r = -sum(math.sin(coeffs[i])*(math.sin((i+1)*coeffs[i]**2/math.pi))**m for i in n)
    return r

# Configure space of hyperparameter
algo_1 = ListTask(is_ordered=True,
                  name = "basinhopping",
                  tasks = ["basinhopping__method", "basinhopping__stepsize"])
algo_2 = ListTask(is_ordered=True,
                  name = "differential_evolution",
                  tasks = ["differential_evolution__strategy",
                           "differential_evolution__popsize"])
start = ChoiceScenario(name = "root", scenarios=[algo_1, algo_2])

# Sampling hyperparameter
list_method = ["BFGS", "Nelder-Mead", "Powell", "L-BFGS-B",
               "TNC", "COBYLA", "SLSQP"]
list_strategy = ["best1bin", "best1exp", "rand1exp", "randtobest1exp",
                 "best2exp", "rand2exp", "randtobest1bin", "best2bin",
                 "rand2bin", "rand1bin"]
sampler = { "basinhopping__stepsize": ([0, 1], "uniform", "float"),
            "basinhopping__method": ([list_method], "choice", "string"),
            "differential_evolution__strategy": ([list_strategy], "choice",
                                                 "string"),
            "differential_evolution__popsize": ([[10, 15, 20, 25, 30]],
                                                "choice", "func")
}

# Evaluation of one configuration
def eval_func(config):
    model, params = config[1]
    x0 = [1.0, 1.0]
    if model == "basinhopping":
        minimizer_kwargs = {"method": params["method"]}
        ret = basinhopping(Michalewicz, x0,
                           stepsize = params["stepsize"],
                           minimizer_kwargs=minimizer_kwargs, niter=20)
        return - (ret.fun / 2)
    elif model == "differential_evolution":
        ret = differential_evolution(Michalewicz, [(0, math.pi), (0, math.pi)],
                               strategy = params["strategy"],
                               popsize = params["popsize"])
        return - (ret.fun / 2)
    raise("Model {0} not implemented yet".format(model))

mosaic = Search(scenario = start, sampler = sampler, eval_func = eval_func)
res = mosaic.run(nb_simulation = 200, generate_image_path = "images")

print(res)
