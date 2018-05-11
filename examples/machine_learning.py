import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import math
import numpy as np
import random

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


from mosaic.mosaic import Search
from mosaic.space import ChildRule
from mosaic.scenario import ListTask, ComplexScenario, ChoiceScenario


# Configure space of hyperparameter
algo_1 = ListTask(is_ordered=False,
                  name = "SVC",
                  tasks = ["SVC__C", "SVC__kernel", "SVC__degree", "SVC__coef0"])
algo_2 = ListTask(is_ordered=True,
                  name = "LogisticRegression",
                  tasks = ["LogisticRegression__penalty",
                           "LogisticRegression__C"])
start = ChoiceScenario(name = "root", scenarios=[algo_1, algo_2])

# Sampling hyperparameter
sampler = { "SVC__C": ([0, 2], "uniform", "float"),
            "SVC__kernel": ([["linear", "poly", "rbf", "sigmoid"]], "choice", "string"),
            "SVC__degree": ([[1, 2, 3, 4, 5]], "choice", "int"),
            "SVC__coef0": ([0., 10.], "uniform", "float"),
            "LogisticRegression__penalty": ([["l1", "l2"]], "choice", "string"),
            "LogisticRegression__C": ([0, 2], "uniform", "float")
}

rules = [ChildRule(applied_to=["SVC__degree"], parent = "SVC__kernel", value = ["poly"]), #Degree only for kernel=poly
         ChildRule(applied_to=["SVC__coef0"], parent = "SVC__kernel", value = ["poly", "sigmoid"])]

# Evaluation of one configuration
def eval_func(config):
    digits = load_digits()
    X, target = digits.data, digits.target

    model, params = config[1]

    if model == "SVC":
        scores = cross_val_score(SVC(**params), X, target, cv = 3)
        return min(scores)
    elif model == "LogisticRegression":
        scores = cross_val_score(LogisticRegression(**params), X, target, cv = 3)
        return min(scores)

mosaic = Search(scenario = start, sampler = sampler, rules = rules,
                eval_func = eval_func, widening_coef = 0.4)
res = mosaic.run(nb_simulation = 500, generate_image_path = "images")

print(res)
