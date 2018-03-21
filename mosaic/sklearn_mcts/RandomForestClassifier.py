import random
import warnings
warnings.simplefilter('default')
import numpy as np

from .sklearn_space import Space_preprocessing
from .sklearn_env import Env_preprocessing

from mosaic.space import Node_space
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

class Space_randomForestClassifier(Space_preprocessing):
	def __init__(self):
		super(Space_randomForestClassifier, self).__init__()

		randomForestClassifier__criterion = Node_space("randomForestClassifier__criterion")
		randomForestClassifier__max_features = randomForestClassifier__criterion.add_child("randomForestClassifier__max_features")
		randomForestClassifier__max_depth = randomForestClassifier__max_features.add_child("randomForestClassifier__max_depth")
		randomForestClassifier__min_samples_split = randomForestClassifier__max_depth.add_child("randomForestClassifier__min_samples_split")
		randomForestClassifier__n_estimators = randomForestClassifier__min_samples_split.add_child("randomForestClassifier__n_estimators")
		#randomForestClassifier__min_samples_leaf = randomForestClassifier__min_samples_split.add_child("randomForestClassifier__min_samples_leaf")
		#randomForestClassifier__min_weight_fraction_leaf = randomForestClassifier__min_samples_leaf.add_child("randomForestClassifier__min_weight_fraction_leaf")
		#randomForestClassifier__max_leaf_nodes = randomForestClassifier__min_weight_fraction_leaf.add_child("randomForestClassifier__max_leaf_nodes")
		#randomForestClassifier__min_impurity_decrease = randomForestClassifier__max_leaf_nodes.add_child("randomForestClassifier__min_impurity_decrease")

		self.append_to_parent(randomForestClassifier__criterion)

		# Add priors later
		self.default_params = {
			"bootstrap": True,
			"n_jobs": -1,
			"warm_start": True,
			"random_state": 42
		}

		self.terminal_pointer = [randomForestClassifier__n_estimators]


class Env_randomForestClassifier(Env_preprocessing):
	def __init__(self, X, y, aggreg_score=np.mean, ressource=2, cv=3):
		super(Env_randomForestClassifier, self).__init__(X, y, aggreg_score)
		self.cv = cv

		sampler = self.space.sampler
		sample_to_add = {
			"randomForestClassifier__criterion": (random.choice, [["gini", "entropy"]]),
			"randomForestClassifier__max_features": (random.choice, [["sqrt", "log2", None]]),
			"randomForestClassifier__max_depth": (random.randint, [1, 50]),
			"randomForestClassifier__min_samples_split": (random.choice, [list(range(2, 10))]),
			"randomForestClassifier__min_samples_leaf": (random.randint, [1, 10]),
			"randomForestClassifier__min_weight_fraction_leaf": (random.uniform, [0, 0.01]),
			"randomForestClassifier__max_leaf_nodes": (random.choice, [[None] + list(range(2, 5))]),
			"randomForestClassifier__min_impurity_decrease": (random.uniform, [0, 0.1]),
			"randomForestClassifier__n_estimators": (random.choice, [[ressource]])
		}
		sampler.update(sample_to_add)

		self.space = Space_randomForestClassifier()
		weight = compute_class_weight("balanced", [0, 1], self.y)
		self.space.default_params["class_weight"] = {0: weight[0], 1:weight[1]}
		self.space.set_sampler(sampler=sampler)

		self.list_estimator["randomForestClassifier"] = RandomForestClassifier
		self.terminal_state = ["randomForestClassifier__n_estimators"]

	def set_ressource(self, val):
		self.space.sampler["randomForestClassifier__n_estimators"] = (random.choice, [[val]])

	def calculate(self, moves):
		return self._calculate(moves)

	def compute_score(self, model, X, y):
		skf = StratifiedKFold(n_splits=self.cv, shuffle=False, random_state=42)

		scores = [0] * self.cv
		i = 0
		for train_index, test_index in skf.split(X, y):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]

			estimator = clone(model)

			with warnings.catch_warnings(record=True) as w:
				estimator.fit(X_train, y_train)

			y_pred = estimator.predict_proba(X_test)[:, 1] # Get proba for y=1
			score = roc_auc_score(y_test, y_pred)

			if score < self.bestscore:
				for j in range(i, 3):
					scores[j] = score
				return scores
			else:
				scores[i] = score

			i += 1
		return scores
