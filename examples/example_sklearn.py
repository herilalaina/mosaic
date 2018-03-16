import sys
import os
import time

PACKAGE_PARENT = '../mcts'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from sklearn_mcts import mcts_preprocessing, mcts_logisticRegression, mcts_lasso, mcts_ridge, mcts_sgdClassifer
from sklearn_mcts import mcts_elasticNet, mcts_linearDiscriminantAnalysis, mcts_kneighborsClassifier, mcts_randomForestClassifier
from sklearn_mcts import mcts_xgboost
from sklearn import datasets

if __name__=="__main__":
	nb_sim = 100

	X, y = datasets.make_classification(n_samples=5000, n_features=500, n_clusters_per_class=10, n_informative=300, random_state=42)
	#X_lasso, y_lasso = datasets.make_classification(n_samples=100, n_features=10000, n_clusters_per_class=1000, n_informative=900, random_state=42, weights=[0.7, 0.3])

	print("MCTS for preprocessing")
	res = mcts_preprocessing(X, y, nb_sim=nb_sim)
	print(res)

	"""print("MCTS for LogisticRegression")
	start = time.time()
	res = mcts_logisticRegression(X, y, nb_sim)
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))

	print("MCTS for Lasso")
	start = time.time()
	res = mcts_lasso(X_lasso, y_lasso, nb_sim)
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))

	print("MCTS for Lasso")
	start = time.time()
	res = mcts_ridge(X_lasso, y_lasso, nb_sim)
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))

	print("MCTS for SGDClassifer")
	start = time.time()
	res = mcts_sgdClassifer(X, y, nb_sim)
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))

	print("MCTS for ElasticNet")
	start = time.time()
	res = mcts_elasticNet(X_lasso, y_lasso, nb_sim)
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))

	print("MCTS for linearDiscriminantAnalysis")
	start = time.time()
	res = mcts_linearDiscriminantAnalysis(X_lasso, y_lasso, nb_sim)
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))

	print("MCTS for kneighborsClassifier")
	start = time.time()
	res = mcts_kneighborsClassifier(X_lasso, y_lasso, nb_sim)
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))

	print("MCTS for randomForestClassifier")
	start = time.time()
	res = mcts_randomForestClassifier(X, y, nb_sim)
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))
	print("MCTS for xgboost")
	start = time.time()
	res = mcts_xgboost(X, y, nb_sim)
	print("Solution after {0} iterations ({1} s): {2}".format(nb_sim, time.time() - start, str(res)))"""
