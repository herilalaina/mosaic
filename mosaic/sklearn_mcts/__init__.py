"""Wrapper for scikit-learn."""

import numpy as np
from mosaic import sklearn_mcts
from mosaic import mcts
from mosaic.sklearn_mcts.sklearn_env import Env_preprocessing
from mosaic.sklearn_mcts import Lasso, ElasticNet
from mosaic.sklearn_mcts.classifier import SGDClassifier, LinearDiscriminantAnalysis, RandomForestClassifier, \
    XGBClassifier, LogisticRegression, KNeighborsClassifier
from mosaic.sklearn_mcts import Ridge

list_env_classifier = {
    "LogisticRegression": LogisticRegression.Env_logisticRegression,
    "Lasso": Lasso.Env_lasso,
    "Ridge": Ridge.Env_ridge,
    "SGDClassifier": SGDClassifier.Env_sgdClassifier,
    "ElasticNet": ElasticNet.Env_elasticNet,
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis.Env_linearDiscriminantAnalysis,
    "KNeighborsClassifier": KNeighborsClassifier.Env_kneighborsClassifier,
    "RandomForestClassifier": RandomForestClassifier.Env_randomForestClassifier,
    "XGBClassifier": XGBClassifier.Env_xgboost
}


def mcts_preprocessing(X, y, nb_sim=200, aggreg_score=np.min):
    sklearn_env = Env_preprocessing(X, y, aggreg_score)
    mcts_model = mcts.MCTS(env=sklearn_env)
    res = mcts_model.run(nb_sim=nb_sim)
    return res


def mcts_model(classifier, X, y, nb_play=25, nb_simulation=20, aggreg_score=np.mean,
                init_ressource=1, init_nb_child=15, nb_step_add_ressource=2, nb_step_to_add_nb_child=1,
                ressource_to_add=5, number_child_to_add=10, cv=3, info=None):

    environement = list_env_classifier[classifier]
    sklearn_env = environement(X, y, aggreg_score, ressource=init_ressource, cv=cv, info=info)
    mcts_model = mcts.MCTS(env=sklearn_env)
    mcts_model.setup(nb_simulation)

    mcts_model.root_node.max_number_child = init_nb_child
    val_resssource = init_ressource
    for i in range(nb_play):
        print("### Play {0}: {1} (Ressource: {2}) ###".format(i, classifier, val_resssource))

        if i % nb_step_add_ressource == 0:
            val_resssource += ressource_to_add
            if val_resssource > 150:
                val_resssource = 150
            mcts_model.env.set_ressource(val_resssource)

        if i % nb_step_to_add_nb_child == 0:
            mcts_model.root_node.max_number_child += number_child_to_add
        yield mcts_model.play()


def mcts_logisticRegression(X, y, nb_sim=200, aggreg_score=np.min):
    sklearn_env = LogisticRegression.Env_logisticRegression(X, y, aggreg_score)
    mcts_model = mcts.MCTS(env=sklearn_env)
    res = mcts_model.run(nb_sim=nb_sim)
    return res


def mcts_lasso(X, y, nb_sim=200):
    sklearn_env = Lasso.Env_lasso(X, y, aggreg_score=np.mean)
    mcts_model = mcts.MCTS(env=sklearn_env)
    res = mcts_model.run(nb_sim=nb_sim)
    return res


def mcts_ridge(X, y, nb_sim=200):
    sklearn_env = Ridge.Env_ridge(X, y, aggreg_score=np.mean)
    mcts_model = mcts.MCTS(env=sklearn_env)
    res = mcts_model.run(nb_sim=nb_sim)
    return res


def mcts_sgdClassifer(X, y, nb_sim=200):
    sklearn_env = SGDClassifier.Env_sgdClassifier(X, y)
    mcts_model = mcts.MCTS(env=sklearn_env)
    res = mcts_model.run(nb_sim=nb_sim)
    return res


def mcts_elasticNet(X, y, nb_sim=200):
    sklearn_env = ElasticNet.Env_elasticNet(X, y, aggreg_score=np.mean)
    mcts_model = mcts.MCTS(env=sklearn_env)
    res = mcts_model.run(nb_sim=nb_sim)
    return res


def mcts_linearDiscriminantAnalysis(X, y, nb_sim=200):
    sklearn_env = LinearDiscriminantAnalysis.Env_linearDiscriminantAnalysis(X, y, aggreg_score=np.mean)
    mcts_model = mcts.MCTS(env=sklearn_env)
    res = mcts_model.run(nb_sim=nb_sim)
    return res


def mcts_kneighborsClassifier(X, y, nb_sim=200):
    sklearn_env = KNeighborsClassifier.Env_kneighborsClassifier(X, y, aggreg_score=np.mean)
    mcts_model = mcts.MCTS(env=sklearn_env)
    res = mcts_model.run(nb_sim=nb_sim)
    return res


def mcts_randomForestClassifier(X, y, nb_sim=200):
    sklearn_env = RandomForestClassifier.Env_randomForestClassifier(X, y)
    mcts_model = mcts.MCTS(env=sklearn_env)
    res = mcts_model.run(nb_sim=nb_sim)
    return res


def mcts_xgboost(X, y, nb_sim=200):
    sklearn_env = XGBClassifier.Env_xgboost(X, y)
    mcts_model = mcts.MCTS(env=sklearn_env)
    res = mcts_model.run(nb_sim=nb_sim)
    return res
