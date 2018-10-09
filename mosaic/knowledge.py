

class Knowledge():
    def __init__(self):
        pass

    def represent_state(self, sequence):
        raise NotImplemented()

    def add(self, state, reward, **kwargs):
        raise NotImplemented()

    def infer(self, state, **kwargs):
        raise NotImplemented()