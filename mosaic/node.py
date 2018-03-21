"""Class Node."""


class Node():
    """Tree node class."""

    max_number_child = 10  # Share for all node

    def __init__(self, state, parent=None):
        """Initialization of Node class."""
        self.visits = 0
        self.reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        """Add child to the current node."""
        child = Node(child_state, self)
        self.children.append(child)

    def update(self, reward):
        """Update node value."""
        self.reward += reward
        self.visits += 1

    def fully_expanded(self):
        """Check if node is fully expanded."""
        tried_children = [c.state for c in self.children]
        new_state = self.state.next_state()

        attempt = 0
        while new_state in tried_children:
            new_state = self.state.next_state()
            if new_state in tried_children:
                if attempt < 5:
                    attempt += 1
                else:
                    return True

        return False

    def __repr__(self):
        """Personalized print of Node class."""
        s = "Name: %s; Nb_visits: %d; reward: %f" % (self.state.getName(),
                                                     self.visits, self.reward)
        return s
