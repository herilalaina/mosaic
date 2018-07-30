class BaseStrategy():
    def __init__(self):
        pass

    def selection(self, parent, ids, vals, visits):
        pass

    def expansion(self, sampler, arg):
        return sampler(*arg)

    def backpropagate(self, value, visit, reward):
        new_val = value + (reward - value) / (visit + 1)
        return new_val, visit + 1

    def playout(self):
        pass


class BaseEarlyStopping():
    def __init__(self):
        pass

    def evaluate(self, func, args):
        return func(*args)
