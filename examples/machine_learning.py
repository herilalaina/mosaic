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

from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn import feature_selection
from sklearn.pipeline import Pipeline

from mosaic.mosaic import Search
from mosaic.space import ChildRule, Parameter
from mosaic.scenario import ListTask, ComplexScenario, ChoiceScenario


# Configure space of hyperparameter
pca = ListTask(is_ordered=False, name = "PCA",
                              tasks = ["PCA__n_components"],
                               rules = [])
selectKBest = ListTask(is_ordered=False, name = "SelectKBest",
                              tasks = [
                                       # "SelectKBest__score_func",
                                       "SelectKBest__k"])
preprocessing = ChoiceScenario(name = "preprocessing", scenarios = [pca, selectKBest])


algo_1 = ListTask(is_ordered=False,
                  name = "SVC",
                  tasks = ["SVC__kernel", "SVC__degree"])
algo_2 = ListTask(is_ordered=True,
                  name = "LogisticRegression",
                  tasks = ["LogisticRegression__penalty",
                           "LogisticRegression__C"])
model = ChoiceScenario(name = "model", scenarios=[algo_1, algo_2])

start = ComplexScenario(name = "root", scenarios=[preprocessing, model], is_ordered=True)

# Sampling hyperparameter
sampler = { "SVC__C": Parameter("SVC__C",[0, 2], "uniform", "float"),
            "SVC__kernel": Parameter("SVC__kernel", ["linear", "poly", "rbf", "sigmoid"], "choice", "string"),
            "SVC__degree": Parameter("SVC__degree", [1, 2, 3], "choice", "int"),
            "LogisticRegression__penalty": Parameter("LogisticRegression__penalty", ["l1", "l2"], "choice", "string"),
            "LogisticRegression__C": Parameter("LogisticRegression__C", [0, 2], "uniform", "float"),
            "PCA__n_components": Parameter("PCA__n_components", [2, 20], "uniform", 'int'),
            "SelectKBest__score_func": Parameter("SelectKBest__score_func", [feature_selection.f_classif, feature_selection.mutual_info_classif, feature_selection.chi2], "choice", "func"),
            "SelectKBest__k": Parameter("SelectKBest__k", [1, 20], "uniform", "int")
}

rules = []

# Evaluation of one configuration
def eval_func(config, bestconfig):
    digits = load_digits()
    X, target = digits.data, digits.target

    preprocessing = None
    classifier = None

    list_available_preprocessing = {
        "PCA": PCA,
        "SelectKBest": SelectKBest,
        "SVC": SVC,
        "LogisticRegression": LogisticRegression
    }

    for name, params in config:
        if name in ["PCA", "SelectKBest"]:
            preprocessing = list_available_preprocessing[name](**params)
        elif  name in ["SVC", "LogisticRegression"]:
            classifier = list_available_preprocessing[name](**params)

    if preprocessing is None or classifier is None:
        raise Exception("Classifier and/or Preprocessing not found\n {0}".format(config))

    pipeline = Pipeline(steps=[("preprocessing", preprocessing), ("classifier", classifier)])
    print(pipeline) # Print algo

    scores = cross_val_score(pipeline, X, target, cv = 3)
    return min(scores)

mosaic = Search(scenario = start, sampler = sampler, rules = rules,
                eval_func = eval_func, widening_coef = 0.5)
res = mosaic.run(nb_simulation = 500, generate_image_path = "images")

print(res)
