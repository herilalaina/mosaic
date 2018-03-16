import sys
import os
import time

PACKAGE_PARENT = '../mcts'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from sklearn_mcts.EnsembleMCTS import EnsembleMTCS
from sklearn import datasets

if __name__=="__main__":
	nb_sim = 100

	X, y = datasets.make_classification(n_samples=5000, n_features=40, n_clusters_per_class=10, n_informative=30, random_state=42)
	list_estimator = ["ElasticNet", "Lasso", "Ridge", "SGDClassifier"]
	model = EnsembleMTCS(list_estimator)
	model.train(X, y)
