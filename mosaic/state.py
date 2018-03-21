"""State of hyperparameter."""

import hashlib


class State():
    """Class state."""
    num_moves = 5  # Maximum move for each state

    def __init__(self, moves=[], env=None, value=0):
        """Initialization."""
        self.moves = moves
        self.value = value
        self.env = env

    def next_state(self):
        """Choose next state."""
        nextmove = self.env.random_state(self.moves)
        next = State(moves=self.moves+[nextmove], env=self.env)
        return next

    def terminal(self):
        """Check is state is terminal."""
        if len(self.moves) == 0:
            return False
        elif self.moves[-1][0] in self.env.terminal_state:
            return True
        return False

    def reward(self):
        """Get reward from state."""
        self.value = self.env.evaluate(self.moves)
        return self.value

    def __str__(self):
        """Personalized print."""
        return str(self.moves[-1][1]).rstrip("\r\n")

    def __hash__(self):
        """Hash state object."""
        return int(hashlib.md5(
                   str(self.moves).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        """Compare two states using hash function."""
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        """Print node."""
        s = "Moves: %s" % (self.moves)
        return s

    def getName(self):
        """Get lastest move."""
        return self.moves[-1]
