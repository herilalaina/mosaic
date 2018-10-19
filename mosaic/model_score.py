from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_is_fitted
import numpy as np


class ScoreModel():
    def __init__(self, nb_param, X=None, y=None):
        self.model = Lasso()
        self.nb_param = nb_param

        if X is not None and y is not None:
            self.X = X
            self.y = y
            self.model.fit(X, y)
        else:
            self.X = []
            self.y = []


    def partial_fit(self, x, y):
        self.X.append(x)
        self.y.append(y)
        self.fit()

    def fit(self):
        self.model.fit(self.X, self.y)

    def importance_variable(self):
        if check_is_fitted(self.model):
            return self.model.coef_
        else:
            raise Exception("ScoreModel not fitted")

    def predict(self, x):
        if check_is_fitted(self.model):
            return self.model.predict(x)
        else:
            raise Exception("ScoreModel not fitted")

    def most_importance_parameter(self, ids):
        if len(self.X) > 1:
            return np.argmax([np.abs(self.model.coef_[id - self.nb_param]) for id in ids])
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
