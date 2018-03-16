class Node():
	max_number_child = 10 # Share for all node

	def __init__(self, state, parent=None):
		self.visits=0
		self.reward=0.0
		self.state=state
		self.children=[]
		self.parent=parent

	def add_child(self,child_state):
		child = Node(child_state,self)
		self.children.append(child)

	def update(self,reward):
		self.reward += reward
		self.visits += 1

	def fully_expanded(self):
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
		s="Name: %s; Nb_visits: %d; reward: %f"%(self.state.getName(), self.visits,self.reward)
		return s
