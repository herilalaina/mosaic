import sys
import os
import time
import pickle
import numpy as np

PACKAGE_PARENT = '../mcts'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from sklearn_mcts import mcts_model
from sklearn_mcts.ElasticNet import Env_elasticNet
from sklearn import datasets
import mcts

if __name__=="__main__":
	nb_sim = 100
	generate_data = True

	# Generate data (expensive!)
	if generate_data:
		print("generate data ...")
		#X, y = datasets.make_classification(n_samples=200, n_features=30000, n_clusters_per_class=100, n_informative=2000, random_state=42, weights=[0.7, 0.3])
		X, y = datasets.make_classification(n_samples=5000, n_features=40, n_clusters_per_class=10, n_informative=30, random_state=42)
		#pickle.dump((X, y), open("data_HD.pickle", "wb"))
	else:
		X, y = pickle.load(open("data_HD.pickle", "rb"))

	print("MCTS for ElasticNet")
	start = time.time()
	for res in mcts_model("ElasticNet", X, y, ressource_to_add=1, number_child_to_add=10, init_nb_child=20, nb_simulation=40):
		pass
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))

	"""print("MCTS for lasso")
	start = time.time()
	res = mcts_model("Lasso", X, y, ressource_to_add=1, number_child_to_add=10, init_nb_child=20, nb_simulation=40)
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))

	print("MCTS for LogisticRegression")
	start = time.time()
	res = mcts_model("LogisticRegression", X, y, ressource_to_add=10, number_child_to_add=10, init_nb_child=10, nb_simulation=20, nb_play=5)
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))

	print("MCTS for RandomForestClassifier")
	start = time.time()
	res = mcts_model("RandomForestClassifier", X, y, ressource_to_add=10, number_child_to_add=5, init_nb_child=15, nb_simulation=30, nb_play=5)
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))

	print("MCTS for Ridge")
	start = time.time()
	res = mcts_model("Ridge", X, y, ressource_to_add=10, number_child_to_add=10, init_nb_child=15, nb_simulation=40, nb_play=5, init_ressource=1)
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))"""

	"""print("MCTS for sgdClassifier")
	start = time.time()
	generator = mcts_model("SGDClassifier", X, y, ressource_to_add=40, number_child_to_add=10, init_nb_child=15, nb_simulation=4, nb_play=50, init_ressource=70)
	for res in generator:
		pass
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))

	print("MCTS for xgboost")
	start = time.time()
	res = mcts_model("XGBClassifier", X, y, ressource_to_add=10, number_child_to_add=10, init_nb_child=15, nb_simulation=4, nb_play=50, init_ressource=1)
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))"""
