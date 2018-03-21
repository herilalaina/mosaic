"""Base environement class."""


class Env():
    """Base class for environement."""

    terminal_state = []

    def __init__(self):
        """Constructor."""
        self.bestscore = -1
        self.history = {}

    def random_state(self, moves):
        """Generate random parameter."""
        pass

    def evaluate(self, moves):
        """Method for moves evaluation."""
        hash_moves = hash(tuple(a for a in moves))

        if hash_moves in self.history:
            return self.history[hash_moves]

        res = self._evaluate(moves)

        # Add into history
        self.history[hash_moves] = res
        return res
