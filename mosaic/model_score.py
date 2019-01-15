from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import numpy as np
import pickle, os


class ScoreModel():
    def __init__(self, nb_param, X=None, y=None):
        self.model = RandomForestRegressor()
        self.nb_param = nb_param
        self.path = path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data_score.p")

        if X is not None and y is not None:
            self.X = X
            self.y = y
            self.model.fit(X, y)
        else:
            X, y = self.load_data()
            X = np.array(X)
            y = np.array(y)
            a = y != 0
            self.X, self.y = X.tolist(), y.tolist()
            self.X, self.y = [], []

        self.nb_added = 0

    def get_mu_sigma_from_rf(self, X):
        list_pred = []
        for estimator in self.model.estimators_:
            x_pred = estimator.predict(X)
            list_pred.append(x_pred)
        return np.mean(list_pred, axis=1), np.std(list_pred, axis=1)

    def load_data(self):
        try:
            return pickle.load(open(self.path, "rb"))
        except:
            return [], []

    def save_data(self):
        pickle.dump((self.X, self.y), open(self.path, "wb"))

    def partial_fit(self, x, y):
        if y > 0:
            self.X.append(x)
            self.y.append(y)
            self.fit()
            # elf.save_data()
            self.nb_added += 1

    def fit(self):
        self.model.fit(self.X, self.y)

    def importance_variable(self):
        if check_is_fitted(self.model):
            return self.model.feature_importances_
        else:
            raise NotFittedError("ScoreModel not fitted")

    def predict(self, x):
        if check_is_fitted(self.model):
            return self.model.predict(x)
        else:
            raise NotFittedError("ScoreModel not fitted")

    def most_importance_parameter(self, ids):
        if self.nb_added > 5:
            weights = [np.abs(self.model.feature_importances_[id - self.nb_param]) for id in ids]
            weights = weights / sum(weights)
            return np.random.choice(list(range(len(ids))), p=weights)
        else:
            return np.random.randint(len(ids))

    def rave_value(self, value, idx, is_categorical, range_value):
        #print(value)
        if len(value) == 1:
            return value[0]
        elif(len(self.X) < 10):
            return np.random.choice(value)

        #TODO: to optimize
        N = len(self.X)
        X_ = np.array(self.X)
        Y_ = np.array(self.y)

        if is_categorical:
            list_value = [0] * len(range_value)

            for v in range(len(range_value)):
                if list_value[v] != 0:
                    list_score = Y_[X_[:, idx - self.nb_param] == v]
                    if len(list_value) > 0:
                        list_value[v] = np.mean(list_value) + np.sqrt(2 * np.log10(N) / len(list_value))
                    else:
                        list_value[v] = 10

            id_max = np.argmax([list_value[v] for v in value])
            return value[id_max]
        else:
            list_value = [0] * len(value)
            res = (X_[:, idx] != 0)
            X = X_[res, :]
            Y = Y_[res]

            if len(Y) > 10:
                sigma = np.std(X[:, idx])
                for i, v in enumerate(value):
                    T = np.sum([np.exp(-(v - x[idx]) ** 2 / (2 * sigma ** 2)) for x, y in zip(X, Y)])
                    if T != 0:
                        list_value[i] = np.sum([np.exp(-(v - x[idx]) ** 2 / (2 * sigma ** 2)) * y / T for x, y in zip(X, Y)])
                    else:
                        list_value[i] = 0
            else:
                return np.random.choice(value)
            id_max = np.argmax(list_value)
            return value[id_max]
