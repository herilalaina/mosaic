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


class EnsembleMTCS():
    def __init__(self, aggreg_score, cv, info):

        self.nb_play = 500
        self.nb_simulation = 2
        self.init_nb_child = 20
        self.number_child_to_add = 1
        self.best_final_score = 0

        self.aggreg_score = aggreg_score
        self.info = info
        self.cv = cv

        if self.info["task"] == "binary.classification":
            self.stacking = LogisticRegression(n_jobs=2)
        elif self.info["task"] == "multiclass.classification":
            self.stacking = LogisticRegression(n_jobs=2)
        else:
            raise Exception("Can't handle task: {0}".format(self.info["task"]))

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

    def train(self, X, y, X_VALID=None, X_TEST=None):
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
        begin_bandit = 1

        for i in range(self.nb_play):
            estimators = []
            is_not_new = []

            choosed_armed = np.argmax(self.bandits)
            print("Choosed: {0}: {1}".format(self.list_model_name[choosed_armed], self.bandits))
            if i < begin_bandit:
                for name in self.list_model_name:
                    p, statut, score = next(models[name])
                    estimators.append(self.create_pipeline(p))
                    is_not_new.append(statut)
            else:
                for index, name in enumerate(self.list_model_name):
                    if index == choosed_armed:
                        self.nb_visits[choosed_armed] += 1
                        p, statut, score = next(models[name])
                        estimators.append(self.create_pipeline(p))
                        is_not_new.append(statut)
                        if statut:
                            reward = 0
                        else:
                            reward = 1
                    else:
                        estimators.append(clone(self.history_model[index][0]))
                        is_not_new.append(True)

            scores = self.cross_validation_estimators(estimators, X, y, is_not_new)
            val_scores = self.aggreg_score(scores)

            print("======> Play {0}: scores: {1} Score:{2}".format(i, scores, val_scores))
            print("======> Best scores: {0}\n\n".format(max(self.best_final_score, val_scores)))
            if i >= begin_bandit:
                self.update_reward(choosed_armed, int(val_scores > self.best_final_score), i)

            if self.best_final_score < val_scores:
                self.best_final_score = val_scores
                yield self.fit_predict(estimators, X, y, X_VALID, X_TEST)
            else:
                self.print_remaining_time()
                yield None, None

    def print_remaining_time(self):
        now = time.time()
        spend = now - self.start_time
        print("\n\n---------------------------------------------------------------")
        print("             Time elapsed: {0}         Remaining time: {1}".format(spend,
                                                                                  float(self.info['time_budget']) - spend))
        print("---------------------------------------------------------------\n\n")

    def update_reward(self, choosed_armed, reward, t):
        self.bandits_mean[choosed_armed] = self.bandits_mean[choosed_armed] + (
                    reward - self.bandits_mean[choosed_armed]) / self.nb_visits[choosed_armed]
        for i in range(len(self.bandits)):
            try:
                self.bandits[i] = self.bandits_mean[i] + math.sqrt(2 * math.log(t) / self.nb_visits[i])
            except:
                self.bandits[i] = 10000

    def fit_predict(self, estimators, X, y, X_VALID, X_TEST):
        train_stack = []
        y_valid = []
        y_test = []

        for e in estimators:
            e.fit(X, y)

            train_stack_ = e.predict_proba(X)
            y_valid_ = e.predict_proba(X_VALID)
            y_test_ = e.predict_proba(X_TEST)

            train_stack.append(train_stack_)
            y_valid.append(y_valid_)
            y_test.append(y_test_)

        train_stack = np.concatenate(train_stack, axis=1)
        y_valid = np.concatenate(y_valid, axis=1)
        y_test = np.concatenate(y_test, axis=1)

        final_valid = []
        final_test = []

        for stacking in self.list_stacking:
            final_valid.append(stacking.predict_proba(y_valid))
            final_test.append(stacking.predict_proba(y_test))

        final_valid = np.mean(final_valid, axis=0)
        final_test = np.mean(final_test, axis=0)

        final_valid[final_valid < 0] = 0
        final_valid[final_valid > 1] = 1
        final_test[final_test < 0] = 0
        final_test[final_test > 1] = 1

        if self.info["task"] == "binary.classification":
            final_valid = final_valid[:, 1]
            final_test = final_test[:, 1]

        return final_valid, final_test

    def cross_validation_estimators(self, estimators, X, y, is_not_new=[]):
        skf = StratifiedKFold(n_splits=self.cv, shuffle=False, random_state=42)
        scores = []

        n_fold = 0

        oof_y = []
        oof_train = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            y_pred = []

            for index, e in enumerate(estimators):

                if is_not_new[index]:
                    e = self.history_model[index][n_fold]
                else:
                    e = clone(e)
                    e.fit(X_train, y_train)
                    if n_fold == 0:
                        self.history_model[index] = [e]
                    else:
                        self.history_model[index].append(e)

                try:
                    y_pred_ = e.predict_proba(X_test)
                except:
                    y_pred_ = e.predict(X_test)

                y_pred.append(y_pred_)

            y_pred = np.concatenate(y_pred, axis=1)
            oof_y.extend(y_test)

            if oof_train == []:
                oof_train = y_pred
            else:
                oof_train = np.concatenate([oof_train, y_pred], axis=0)

            n_fold += 1

        oof_train = np.array(oof_train)
        oof_y = np.array(oof_y)

        skf = StratifiedKFold(n_splits=self.cv, shuffle=False, random_state=43)
        scores = []

        self.list_stacking = []

        for train_index, test_index in skf.split(oof_train, oof_y):
            e = clone(self.stacking)
            X_train, X_test = oof_train[train_index], oof_train[test_index]
            y_train, y_test = oof_y[train_index], oof_y[test_index]

            with warnings.catch_warnings(record=True) as w:
                e.fit(X_train, y_train)

            self.list_stacking.append(e)

            score = sklearn_mcts.calculate_score_metric(e, X_test, y_test, self.info)

            scores.append(score)

        return scores

    def create_pipeline(self, res):
        pipeline = []
        preprocessing = res["preprocessing"]
        if preprocessing != None:
            pipeline.append(("preprocessing", preprocessing))

        pipeline.append(("estimator", clone(res["estimator"])))
        return Pipeline(pipeline)
