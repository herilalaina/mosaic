from ..space import Node_space
from ..space import Space_sklearn_preprocessing

def test_space_sklearn():
    root = Node_space("root")

    pca = root.add_child("pca")
    pca__n_components = pca.add_child("n_components")

    selectkbest = root.add_child("selectkbest")
    selectkbest__score_func = selectkbest.add_child("score_func")
    selectkbest__k = selectkbest__score_func.add_child("k")

    assert(set(["pca", "selectkbest"]) == set(root.children.keys()))
    assert(set(["root"]) == set(pca.parent.keys()))
    assert(set(["root"]) == set(selectkbest.parent.keys()))
    assert(set(["score_func"]) == set(selectkbest__k.parent.keys()))
    assert(set(["k"]) == set(selectkbest__score_func.children.keys()))

    svc = Node_space("svc")
    svc__C = svc.add_child("svc__C")
    svc__kernel = svc__C.add_child("svc__kernel")

    lr = Node_space("logisticRegression")
    lr__penalty = lr.add_child("penalty")
    lr__C = lr__penalty.add_child("C")

    pca__n_components.append_childs([svc, lr])
    selectkbest__k.append_childs([svc, lr])

    assert(set(["svc", "logisticRegression"]) == set(pca__n_components.children.keys()))
    assert(set(["n_components", "k"]) == set(svc.parent.keys()))
    assert(set(["n_components", "k"]) == set(lr.parent.keys()))

def test__space_sklearn_preprocessing():
    space = Space_sklearn_preprocessing()
    assert(set(space.get_child("root")) == set(["PCA", "selectKBest"]))
    assert(set(space.get_child("PCA")) == set(["PCA__n_components"]))
    assert(set(space.get_child("selectKBest")) == set(["selectKBest__score_func"]))
    assert(set(space.get_child("selectKBest__score_func")) == set(["selectKBest__k"]))
