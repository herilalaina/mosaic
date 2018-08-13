import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn import feature_selection
from sklearn.pipeline import Pipeline

from mosaic.mosaic import Search
from mosaic.simulation.parameter import Parameter
from mosaic.simulation.scenario import ImportanceScenarioStatic


# Configure space of hyperparameter

graph = {
    "root": ["PCA", "SelectKBest"],
    "PCA": ["PCA__n_components"],
    "PCA__n_components": ["SVC", "LogisticRegression"],
    "SelectKBest": ["SelectKBest__k"],
    "SelectKBest__k": ["SelectKBest__score_func"],
    "SelectKBest__score_func": ["SVC", "LogisticRegression"],

    # Algo
    "SVC": ["SVC__kernel"],
    "LogisticRegression": ["LogisticRegression__penalty"]
}

start = ImportanceScenarioStatic(graph)

# Sampling hyperparameter
sampler = { "SVC__C": Parameter("SVC__C",[0, 2], "uniform", "float"),
            "SVC__kernel": Parameter("SVC__kernel", ["linear", "rbf", "sigmoid"], "choice", "string"),
            "LogisticRegression__penalty": Parameter("LogisticRegression__penalty", ["l1", "l2"], "choice", "string"),
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
                eval_func = eval_func)
res = mosaic.run(nb_simulation = 500, generate_image_path = "images")

print(res)
