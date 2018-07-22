class BaseStrategy():
    def __init__(self):
        pass

    def selection(self, parent, vals, visits):
        pass

    def expansion(self):
        pass

    def backpropagate(self, value, visit, reward):
        new_val = value + (reward - value) / (visit + 1)
        return new_val, visit + 1

    def playout(self):
        pass