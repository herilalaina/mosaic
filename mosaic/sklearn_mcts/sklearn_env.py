# Model
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import SVC
from sklearn.random_projection import GaussianRandomProjection
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier

# Model scoring
from sklearn.model_selection import cross_val_score

# preprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

# utils
import numpy as np
import warnings
from sklearn.pipeline import Pipeline


from mosaic.env import Env
from mosaic.sklearn_mcts.sklearn_space import Space_preprocessing

import random

class Env_sklearn(Env):
	history_preprocessing = {}

	def __init__(self, aggreg_score=np.min):
		super(Env_sklearn, self).__init__()
		self.list_estimator = {}
		self.aggreg_score = aggreg_score
		self.have_preprocessing = True
		self.count_evaluation = 0

	def _evaluate(self, moves):
		scores, estimator = self.calculate(moves)
		val_score = self.aggreg_score(scores)
		if val_score > self.bestscore:
			self.bestscore = self.aggreg_score(scores)
			print("New best score: -> {0}: Reward {1}".format(scores, val_score))
			self.best_model = estimator
		self.count_evaluation += 1
		return val_score

	def _calculate(self, moves):
		"""if self.count_evaluation % 3 == 0:
			print("{0}\t :-)".format(self.count_evaluation), end='\r')
		elif self.count_evaluation % 3 == 1:
			print("{0}\t :-*".format(self.count_evaluation), end='\r')
		else:
			print("{0}\t :-D".format(self.count_evaluation), end='\r')"""

		# Prepare pipeline
		pipeline = self.parse_params(moves)
		pipeline[-1][1].set_params(**self.space.default_params)

		# Calculate scores
		try:
			if self.have_preprocessing:
				estimator = pipeline[1][1]
				new_data, preprocessing, error = self.preprocess(pipeline)
				if error:
					return [0, 0, 0], {"preprocessing": preprocessing, "estimator": estimator}
			else:
				new_data = self.X
				preprocessing = None
				estimator = pipeline[0][1]

			scores = self.compute_score(estimator, new_data, self.y)
		except Exception as e:
			raise(e)
			return [0, 0, 0], {"preprocessing": preprocessing, "estimator": estimator}

		return scores, dict({"preprocessing": preprocessing, "estimator": estimator})

	def preprocess(self, pipeline):
		ident_prep = hash(pipeline[0])
		if ident_prep in self.history_preprocessing:
			preprocessing = self.history_preprocessing[ident_prep]
		else:
			preprocessing = pipeline[0][1]
			try:
				with warnings.catch_warnings(record=True) as w:
					preprocessing.fit(self.X, self.y)
			except:
				return self.X, None, 1
			self.history_preprocessing[ident_prep] = preprocessing

		new_data = preprocessing.transform(self.X)
		return new_data, preprocessing, 0

	def random_state(self, moves):
		return self.space.sample(moves[-1][0], moves=moves)

	def get_estimator(self, name):
		if name in self.list_estimator:
			return self.list_estimator[name]
		else:
			return name

	def parse_params(self, moves):
		list_estimator_params = {}
		pipeline = []

		for name, value in moves[1:]:
			l_name = name.split("__")
			if len(l_name) == 1 or pipeline[-1][0] != l_name[0]:
				name_estimator = l_name[0]
				pipeline.append([name_estimator, self.get_estimator(name_estimator)()])
			else:
				name_estimator, estimator_param = l_name[0], l_name[1]

				params = {estimator_param : self.get_estimator(value)}
				pipeline[-1][1].set_params(**params)

		return [(a[0],a[1]) for a in pipeline]


class Env_preprocessing(Env_sklearn):
	def __init__(self, X, y, aggreg_score=np.mean):
		super(Env_preprocessing, self).__init__(aggreg_score)

		dims = np.linspace(5, min(X.shape[1], 500), 10).astype(int)
		sampler = {
			"PCA__n_components": (random.choice, [dims]),
			"selectKBest__score_func": (random.choice, [["f_classif", "f_regression"]]),
			"selectKBest__k": (random.choice, [dims]),
			"gaussianRandomProjection__n_components": (random.choice, [dims]),
			"latentDirichletAllocation__n_components": (random.choice, [dims]),
			"latentDirichletAllocation__learning_decay": (random.choice, [np.random.uniform(0.5, 1, 10)])
		}

		self.list_estimator = {
			"PCA": IncrementalPCA,
			"selectKBest": SelectKBest,
			"f_classif": f_classif,
			"f_regression": f_regression,
			"identity": FunctionTransformer,
			"gaussianRandomProjection": GaussianRandomProjection,
			"latentDirichletAllocation": LatentDirichletAllocation
		}

		self.terminal_state = ["selectKBest__k", "PCA__n_components", "identity", "gaussianRandomProjection__n_components", "latentDirichletAllocation__learning_decay"]
		self.space = Space_preprocessing()
		self.space.set_sampler(sampler=sampler)
		self.X = X
		self.y = y
		self.best_model = None


	def random_state(self, moves):
		return self.space.sample(moves[-1][0])

	def calculate(self, moves):
		# Prepare
		name_estimator = moves[1][0]
		Estimator = self.get_estimator(name_estimator)
		params = self._parse_params(moves)
		estimator = Estimator()
		estimator.set_params(**params)

		# Add into pipeline
		pipeline = [(name_estimator, estimator),
					("passiveAggressiveClassifier", PassiveAggressiveClassifier(warm_start=True, class_weight="balanced", n_jobs=-1, max_iter=50))]
		model = Pipeline(pipeline)

		# Calculate scores
		try:
			scores = cross_val_score(model, self.X, self.y, n_jobs=1, scoring="roc_auc")
		except Exception as e:
			print(e)
			return tuple([[0, 0, 0], estimator])

		return scores, estimator

	def _parse_params(self, params):
		list_param = {}
		for name, value in params[2:]:
			_name = name.split("__")[1]
			list_param[_name] = self.get_estimator(value)
		return list_param
