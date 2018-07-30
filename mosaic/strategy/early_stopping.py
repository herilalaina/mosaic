from mosaic.strategy import BaseEarlyStopping

class Hyperband(BaseEarlyStopping):
    def __init__(self):
        super().__init__()

    def evaluate(self):
        pass
