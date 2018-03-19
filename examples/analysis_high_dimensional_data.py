"""Example of high dimensional data."""

import sys
import time
import pickle

from sklearn import datasets

sys.path.append('mosaic')
from sklearn_mcts import mcts_model  # noqa

if __name__ == "__main__":
    nb_sim = 100
    generate_data = False

    # Generate data (expensive!)
    if generate_data:
        print("generate data ...")
        """ X, y = datasets.make_classification(n_samples=200, n_features=30000,
        n_clusters_per_class=100, n_informative=2000, random_state=42,
        weights=[0.7, 0.3])"""
        X, y = datasets.make_classification(n_samples=5000,
                                            n_features=40,
                                            n_clusters_per_class=10,
                                            n_informative=30,
                                            random_state=42)
        pickle.dump((X, y), open("data_HD.pickle", "wb"))
    else:
        X, y = pickle.load(open("data_HD.pickle", "rb"))

    print("MCTS for ElasticNet")
    start = time.time()
    for res in mcts_model("ElasticNet", X, y, ressource_to_add=1,
                          number_child_to_add=10,
                          init_nb_child=20, nb_simulation=40):
        pass
    print("Solution after {0} iterations ({1} s): {2}"
          .format(nb_sim, time.time() - start, str(res)))

    print("MCTS for lasso")
    start = time.time()
    res = mcts_model("Lasso", X, y, ressource_to_add=1, number_child_to_add=10,
                     init_nb_child=20, nb_simulation=40)
    print("Solution after {0} iterations ({1} s): {2}"
          .format(nb_sim, time.time() - start, str(res)))
