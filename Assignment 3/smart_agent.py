#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mar 30, 10:13:15
@last modified : 2021 Apr 19, 19:07:34
"""


import numpy as np
from core.player import Player, Color
from seega.seega_rules import SeegaRules
from copy import deepcopy
from time import perf_counter
from collections import defaultdict

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class AI(Player):

	in_hand = 12
	score = 0
	name = "Smart Agent"

	def __init__(self, color):
		super(AI, self).__init__(color)
		self.position = color.value
		self._remaining_time = float('inf')
		self._taken_times = defaultdict(list)
		self._total_time = None

	def play(self, state, remain_time):
		print("")
		print(f"Player {self.position} is playing.")
		print("time remain is ", remain_time, " seconds")
		if self._total_time is None:
			self._total_time = remain_time
		self._remaining_time = remain_time

		def depth_policy():
			return 25 if state.phase == 2 else 1

		def time_policy():
			MAX_FRAC = 1/200
			MIN_FRAC = 1/2000
			MIN, MAX = self._total_time*MIN_FRAC, self._total_time*MAX_FRAC

			alpha = self._alpha_winning(state)
			logging.info(f"Current alpha : {alpha}")
			winning_time = MIN + (1 - alpha) * (MAX -MIN)

	
			alpha_time = self._remaining_time / self._total_time
			schedulers = dict(
					linear=MIN + alpha_time*(MAX - MIN), # TODO
					)

			return schedulers['linear']



		
		self._running_depth = 0
		self._max_depth = depth_policy()
		self._max_running_time = time_policy()

		self._start_minimax = perf_counter()		
		best_action = minimax_search(state, self)

		self._taken_times[self._running_depth].append(self._running_time)

		return best_action

	@property
	def _running_time(self):
		return perf_counter() - self._start_minimax 

	def _alpha_winning(self, state):
		return .5 + .5 * (state.score[self.position] - state.score[-self.position]) / state.MAX_SCORE

	@property
	def _alpha_time(self):
		return self._remaining_time / self._total_time
	def successors(self, state):
		"""successors.
		The successors function must return (or yield) a list of pairs (a, s) in which a is the action played to reach the state s.

		:param state: the state for which we want the successors
		"""
		for action in SeegaRules.get_player_actions(state, self.position):
			try:
				next_state = deepcopy(state)
			except Exception:
				import pdb; pdf.set_trace()
			SeegaRules.act(next_state, action, self.position)
			yield action, next_state

	def cutoff(self, state, depth):
		"""cutoff.
		The cutoff function returns true if the alpha-beta/minimax search has to stop and false otherwise.

		:param state: the state for which we want to know if we have to apply the cutoff
		:param depth: the depth of the cutoff
		"""
		def timing_cutoff():
			return self._running_time > self._max_running_time 

		def depth_cutoff():
			return depth > self._max_depth

		is_cutoff = False

		# Check if the game is at the end
		is_cutoff |= SeegaRules.is_end_game(state)

		# Get the cutoff from the current depth
		is_cutoff |= depth_cutoff()

		# Get the current cutoff from the time running the minimax
		is_cutoff |= timing_cutoff()

		self._running_depth = max(self._running_depth, depth)

		return is_cutoff

	def evaluate(self, state):
		"""evaluate.
		The evaluate function must return an integer value representing the utility function of the board.

		:param state: the state for which we want the evaluation scalar
		"""
		def player_wins(player):
			return state.score[player] == state.MAX_SCORE

		score = 0.0

		# Self score
		score += state.score[self.position]

		# Oponent score
		# score -= state.score[-self.position]

		# Winning state
		# score += 100 if player_wins(self.position) else 0
		# score -= 100 if player_wins(-self.position) else 0

		return score

	def set_score(self, new_score):
		self.score = new_score

	def update_player_infos(self, infos):
		self.in_hand = infos["in_hand"]
		self.score = infos["score"]

	def reset_player_informations(self):
		self.in_hand = 12
		self.score = 0


"""
MiniMax and AlphaBeta algorithms.
Adapted from:
	Author: Cyrille Dejemeppe <cyrille.dejemeppe@uclouvain.be>
	Copyright (C) 2014, Universite catholique de Louvain
	GNU General Public License <http://www.gnu.org/licenses/>
"""

inf = float("inf")


def minimax_search(state, player, prune=True):
	"""Perform a MiniMax/AlphaBeta search and return the best action.

	Arguments:
	state -- initial state
	player -- a concrete instance of class AI implementing an Alpha-Beta player
	prune -- whether to use AlphaBeta pruning

	"""

	def max_value(state, alpha, beta, depth):
		if player.cutoff(state, depth):
			return player.evaluate(state), None
		val = -inf
		action = None
		for a, s in player.successors(state):
			if (
				s.get_latest_player() == s.get_next_player()
			):  # next turn is for the same player
				v, _ = max_value(s, alpha, beta, depth + 1)
			else:  # next turn is for the other one
				v, _ = min_value(s, alpha, beta, depth + 1)
			if v > val:
				val = v
				action = a
				if prune:
					if v >= beta:
						return v, a
					alpha = max(alpha, v)
		return val, action

	def min_value(state, alpha, beta, depth):
		if player.cutoff(state, depth):
			return player.evaluate(state), None
		val = inf
		action = None
		for a, s in player.successors(state):
			if (
				s.get_latest_player() == s.get_next_player()
			):  # next turn is for the same player
				v, _ = min_value(s, alpha, beta, depth + 1)
			else:  # next turn is for the other one
				v, _ = max_value(s, alpha, beta, depth + 1)
			if v < val:
				val = v
				action = a
				if prune:
					if v <= alpha:
						return v, a
					beta = min(beta, v)
		return val, action

	_, action = max_value(state, -inf, inf, 0)
	return action
