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
