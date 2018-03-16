#!/usr/bin/env python
import random
import math
import hashlib
import logging
import argparse
from policy import UCT
from node import Node
from state import State
from env import Env

"""
Monte Carlo Tree Search implementation.
"""

class MCTS():
	def __init__(self, env, logfile='mcts.log'):
		self.env = env
		self.root_node = Node(State(env=env,moves=[("root", None)]))

		# Set up logger
		self.logger = logging.getLogger('mcts')
		hdlr = logging.FileHandler('mcts.log', mode='w')
		formatter = logging.Formatter('%(asctime)s %(message)s')
		hdlr.setFormatter(formatter)
		self.logger.addHandler(hdlr)
		self.logger.setLevel(logging.DEBUG)

		# Policy
		self.policy = UCT()

	def MCT_SEARCH(self, budget, root):
		for iter in range(int(budget)):
			front = self.TREEPOLICY(root)
			reward = self.random_policy(front.state)
			self.BACKUP(front, reward)
		return self.policy.BESTCHILD(root, 0)

	def TREEPOLICY(self, node):
		SCALAR = 1 / math.sqrt(2.0)
		while node.state.terminal()==False:
			if len(node.children)==0:
				return self.EXPAND(node)
			elif random.uniform(0,1) <0.1:
				node = self.policy.BESTCHILD(node, SCALAR)
			else:
				if node.fully_expanded()==False:
					return self.EXPAND(node)
				else:
					node = self.policy.BESTCHILD(node, SCALAR)
		return node

	def EXPAND(self, node):
		tried_children = [c.state for c in node.children]
		new_state = node.state.next_state()
		while new_state in tried_children:
			new_state = node.state.next_state()
		node.add_child(new_state)
		return node.children[-1]

	def random_policy(self, state):
		while state.terminal() == False:
			state=state.next_state()
		return state.reward()

	def best_decision(self):
		node = self.root_node
		list_node = []
		while node.state.terminal()==False:
			node = self.policy.BESTCHILD(node, 0)
			list_node.append(node.state.moves[-1])
		return list_node

	def BACKUP(self, node, reward):
		while node!=None:
			node.visits += 1
			node.reward += reward
			node=node.parent

	def setup(self, nb_sim_each_play):
		self.nb_sim_each_play = nb_sim_each_play

	def play(self, info=""):
		self.logger.debug("--------------------------------")
		self.logger.debug("Play {0}".format(info))
		current_node = self.root_node
		l = 0

		old_best_model = self.env.best_model

		while (len(current_node.children) > 0) or (l == 0):
			self.logger.debug("--------------------------------")
			current_node_ = self.MCT_SEARCH(self.nb_sim_each_play, current_node)
			self.logger.debug("level %d" % l)
			self.logger.debug("Num Children: %d" % len(current_node.children))
			for i,c in enumerate(current_node.children):
				self.logger.debug("\t {0} : {1}".format(i,c))
			current_node = current_node_
			l += 1

		return self.env.best_model, old_best_model == self.env.best_model, self.env.bestscore

	def run(self, nb_sim):
		current_node = self.root_node
		l = 0
		while (len(current_node.children) > 0) or (l == 0):
			current_node_ = self.MCT_SEARCH(nb_sim, current_node)
			self.logger.debug("level %d" % l)
			self.logger.debug("Num Children: %d" % len(current_node.children))
			for i,c in enumerate(current_node.children):
				self.logger.debug("\t {0} : {1}".format(i,c))
			self.logger.debug("Best Child: {0}, value: {1}".format(current_node_.state, current_node_.state.value))
			current_node = current_node_
			self.logger.debug("--------------------------------")
			l += 1
		return self.env.best_model
