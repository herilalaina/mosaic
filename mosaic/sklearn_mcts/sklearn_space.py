import random

from mosaic.space import Space, Node_space


class Space_sklearn(Space):
    def __init__(self):
        self.root = Node_space("root")


class Space_preprocessing(Space_sklearn):
    def __init__(self):
        super(Space_preprocessing, self).__init__()

        pca = self.root.add_child("PCA")
        pca__n_components = pca.add_child(child_name="PCA__n_components")

        selectkbest = self.root.add_child("selectKBest")
        selectkbest__score_func = selectkbest.add_child("selectKBest__score_func")
        selectkbest__k = selectkbest__score_func.add_child("selectKBest__k")

        identity = self.root.add_child("identity")

        latentDirichletAllocation = self.root.add_child("latentDirichletAllocation")
        latentDirichletAllocation__n_components = latentDirichletAllocation.add_child(
            "latentDirichletAllocation__n_components")
        latentDirichletAllocation__learning_decay = latentDirichletAllocation__n_components.add_child(
            "latentDirichletAllocation__learning_decay")

        self.terminal_pointer = [selectkbest__k, pca__n_components, identity, latentDirichletAllocation__learning_decay]

    def append_to_parent(self, node):
        for n in self.terminal_pointer:
            n.append_child(node)
