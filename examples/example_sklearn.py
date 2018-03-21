"""MOSAIC applied to sklearn."""
import sys
sys.path.append('../')
import time
from sklearn import datasets
from mosaic import sklearn_mcts

if __name__ == "__main__":
    nb_sim = 100

    X, y = datasets.make_classification(n_samples=5000, n_features=500,
                                        n_clusters_per_class=10,
                                        n_informative=300, random_state=42)

    print("MCTS for preprocessing")
    res = sklearn_mcts.mcts_preprocessing(X, y, nb_sim=nb_sim)
    print(res)

    print("MCTS for LogisticRegression")
    start = time.time()
    res = sklearn_mcts.mcts_logisticRegression(X, y, nb_sim)
    print("Solution after {0} iterations ({1} s): {2}"
          .format(nb_sim, time.time() - start, str(res)))
