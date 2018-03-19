"""Example of ensemble."""

import sys
from sklearn import datasets

sys.path.append("mosaic")
from sklearn_mcts.EnsembleMCTS import EnsembleMTCS  # noqa

if __name__ == "__main__":
    nb_sim = 100

    X, y = datasets.make_classification(n_samples=5000, n_features=40,
                                        n_clusters_per_class=10,
                                        n_informative=30, random_state=42)
    list_estimator = ["ElasticNet", "Lasso", "Ridge", "SGDClassifier"]
    model = EnsembleMTCS(list_estimator)
    model.train(X, y)
