import numpy as np
import warnings
import time
import math
import random

warnings.filterwarnings("ignore")

from sklearn.base import clone
from mosaic import sklearn_mcts
from mosaic.sklearn_mcts import mcts_model
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression


class VanillaMOSAIC():
    def __init__(self, aggreg_score, cv, info):

        self.nb_play = 500
        self.nb_simulation = 2
        self.init_nb_child = 20
        self.number_child_to_add = 1
        self.best_final_score = 0

        self.aggreg_score = aggreg_score
        self.info = info
        self.cv = cv

        self.best_estimators = None

    def define_strategy(self, X):
		self.list_estimator = ["ElasticNet", "Lasso", "Ridge", "SGDClassifier", "LogisticRegression", "RandomForestClassifier", "XGBClassifier"]
		self.acceleration = 5
		self.cv = 3
		if X.shape[0] * 2 < X.shape[1]:
			self.list_estimator = ["ElasticNet", "Lasso", "Ridge", "RandomForestClassifier", "XGBClassifier"]
			self.cv = 10
			self.ressource_to_add = 1
			self.nb_step_add_ressource = 1
			self.init_ressource = 1
		elif X.shape[0] < 6000:
			self.ressource_to_add = 20
			self.nb_step_add_ressource = 20
			self.init_ressource = 100
		else:
			self.acceleration = 1
			self.ressource_to_add = 5
			self.nb_step_add_ressource = 5
			self.init_ressource = 5
        self.start_time = time.time()
        self.history_model = {}
        self.bandits = [1000] * len(list_model_name)
        self.bandits_mean = [0] * len(list_model_name)
        self.nb_visits = [0] * len(list_model_name)

    def fit(self, X, y):
        if X.shape[0] * 2 < X.shape[1]:
            high_dimensional_data = True
        else:
            high_dimensional_data = False

        self.define_strategy(X)

        models = {}
        for name in self.list_model_name:
            if name in ["RandomForestClassifier", "XGBClassifier", "LogisticRegression", "SGDClassifier"]:
                models[name] = mcts_model(name, X, y, self.nb_play, self.nb_simulation, self.aggreg_score,
                                          self.init_ressource, self.init_nb_child, self.nb_step_add_ressource,
                                          self.nb_step_to_add_nb_child,
                                          self.ressource_to_add, self.number_child_to_add, self.cv, self.info)
            else:
                models[name] = mcts_model(name, X, y, self.nb_play, self.nb_simulation * self.acceleration,
                                          self.aggreg_score,
                                          self.init_ressource, self.init_nb_child, self.nb_step_add_ressource,
                                          self.nb_step_to_add_nb_child,
                                          self.ressource_to_add, self.number_child_to_add, self.cv, self.info)

        for i in range(self.nb_play):
            choosed_armed = np.argmax(self.bandits)
            print("Choosed: {0}: {1}".format(self.list_model_name[choosed_armed], self.bandits))

            for index, name in enumerate(self.list_model_name):
                if index == choosed_armed:
                    self.nb_visits[choosed_armed] += 1
                    p, statut, score = next(models[name])
                    found_estimator = self.create_pipeline(p))
                    if statut:
                        reward = 0
                    else:
                        reward = 1

                    if score > self.best_final_score:
                        self.best_final_score = score
                        self.best_estimators = found_estimator

            self.update_reward(choosed_armed, reward, i)

    def refit(self, X, y):
        self.best_estimators.fit(X, y)

    def predict_proba(self, X_test):
        return self.best_estimators.predict_proba(X_test)

    def update_reward(self, choosed_armed, reward, t):
        self.bandits_mean[choosed_armed] = self.bandits_mean[choosed_armed] + (
                    reward - self.bandits_mean[choosed_armed]) / self.nb_visits[choosed_armed]
        for i in range(len(self.bandits)):
            try:
                self.bandits[i] = self.bandits_mean[i] + math.sqrt(2 * math.log(t) / self.nb_visits[i])
            except:
                self.bandits[i] = 10000

    def create_pipeline(self, res):
        pipeline = []
        preprocessing = res["preprocessing"]
        if preprocessing != None:
            pipeline.append(("preprocessing", preprocessing))

        pipeline.append(("estimator", clone(res["estimator"])))
        return Pipeline(pipeline)
