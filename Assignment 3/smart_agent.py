#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mar 30, 10:13:15
@last modified : 2021 Apr 22, 10:22:41
"""


import numpy as np
from core.player import Player, Color
from seega.seega_rules import SeegaRules
from seega.seega_actions import SeegaAction
from copy import deepcopy
from time import perf_counter
from collections import defaultdict

import math
import logging

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)


class AI(Player):

    in_hand = 12
    score = 0
    name = "Smart Agent"
    _running_depth = 0  # Track the maximum depth in the iterative deepening
    _max_depth = -666

    def __init__(self, color):
        super(AI, self).__init__(color)
        self.color = color
        self.oponent = Color.green if Color.green == color else Color.black
        self.position = color.value
        self._remaining_time = float("inf")
        self._total_time = None
        self._tracking_list = defaultdict(lambda: 0)

    def play(self, state, remain_time):
        print("")
        print(f"Player {self.position} is playing.")
        print("time remain is ", remain_time, " seconds")

        if self._total_time is None:
            self._total_time = remain_time
        self._remaining_time = remain_time

        def time_policy(policy="exponential", min_frac=1 / 2000, max_frac=1 / 100):
            """time_policy.

            :param policy: the desired policy between ('linear', 'exponential')
            :param min_frac: the minimum fraction of total time allowed for computing time
            :param max_frac: the maximum fraction of total time allowed for computing time
            """
            min, max = self._total_time * min_frac, self._total_time * max_frac

            # Calculate the linear and exponential policy
            logging.info(f"alpha time : {self._alpha_time}")
            schedulers = dict(
                linear=min + self._alpha_time * (max - min),
                exponential=min
                + (np.exp(self._alpha_time * math.log(2)) - 1) * (max - min),
            )

            return schedulers[policy]

        self._max_running_time = time_policy("exponential")
        print(self._max_running_time)

        # Begining the search
        self._start_minimax = perf_counter()
        best_action = self.iterative_deepening(state)

        tracked_state = deepcopy(state)
        SeegaRules.act(tracked_state, best_action, self.position)
        self._track_state(tracked_state)

        return best_action

    @property
    def _running_time(self):
        return perf_counter() - self._start_minimax

    @property
    def _alpha_time(self):
        return self._remaining_time / self._total_time

    def _alpha_winning(self, state):
        """_alpha_winning.

        less than 0.5 means loosing
        greater than 0.5 means winning
        :param state: the current state
        """
        return (
            0.5
            + 0.5
            * (state.score[self.position] - state.score[-self.position])
            / state.MAX_SCORE
        )

    def successors(self, state):
        """successors.
        The successors function must return (or yield) a list of pairs (a, s) in which a is the action played to reach the state s.

        :param state: the state for which we want the successors
        """
        next_player = state.get_next_player()
        is_our_turn = next_player == self.position
        successors = list()

        if not is_our_turn and self._already_tracked(state):
            return list()

        for action in SeegaRules.get_player_actions(state, next_player):
            next_state = deepcopy(state)
            if SeegaRules.act(next_state, action, next_player):
                successors.append((action, next_state))

        # Sort states with their evaulation values (reverse : min/max)
        successors.sort(key=lambda x: self.evaluate(x[1]), reverse=not is_our_turn)
        # logging.info(f'Next states for {["oponent", "us"][is_our_turn]} -> {successors}')

        # Get all not already tracked states if loosing and is our turn
        if is_our_turn and self._alpha_winning(state) < 0.5:
            not_tracked = list(
                filter(lambda elem: not self._already_tracked(elem[1]), successors)
            )
            if not_tracked:
                successors = not_tracked

        return successors

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

        # Track the maximum depth
        self._running_depth = max(self._running_depth, depth)

        return is_cutoff

    def evaluate(self, state):
        """evaluate.
        The evaluate function must return an integer value representing the utility function of the board.

        :param state: the state for which we want the evaluation scalar
        """
        cell_groups = dict(
            center=(2, 2),
            star_center=[(2, 1), (2, 3), (1, 2), (3, 2)],
            square_center=[(1, 1), (1, 3), (3, 1), (3, 3)],
            star_ext=[(2, 0), (2, 4), (0, 2), (4, 2)],
            square_ext=[(0, 0), (0, 4), (4, 0), (4, 4)],
        )

        def player_wins(player):
            return state.score[player] == state.MAX_SCORE

        def border_gravity(player):
            def is_border(cell):
                x, y = cell
                m, n = state.board.board_shape
                return x == 0 or y == 0 or x == m - 1 or y == n - 1

            return sum(
                list(
                    map(
                        is_border,
                        state.board.get_player_pieces_on_board(
                            Color.green if Color.green.value == player else Color.black
                        ),
                    )
                )
            )

        def evaluate_cells(color):
            def is_player_cell(cell, color):
                return state.board.get_cell_color(cell) == color

            def direct_center_score():
                score = 0.0
                for base_cell in cell_groups["star_center"]:
                    for oponent_cell, ext_cell in zip(
                        cell_groups["star_center"], cell_groups["star_ext"]
                    ):
                        if (
                            is_player_cell(base_cell, color)
                            and is_player_cell(oponent_cell, -color)
                            and is_player_cell(ext_cell, color)
                        ):
                            score += 1
                return score

            score = 0.0

            if state.phase == 1:
                score += direct_center_score()
            else:
                score += is_player_cell(cell_groups["center"], color)

            return score

        score = 0.0

        if state.phase == 1:

            score += evaluate_cells(self.position)
            score -= evaluate_cells(-self.position)

        elif state.phase == 2:

            # Self score
            score += state.score[self.position]
            score -= state.score[-self.position]

            score += evaluate_cells(self.color)
            score -= evaluate_cells(self.oponent)

            # score += border_gravity(self.position) / state.MAX_SCORE
            # score -= border_gravity(-self.position) / state.MAX_SCORE

            # Winning state
            if SeegaRules.is_end_game(state):
                score += 100 if self._alpha_winning(state) > 0.5 else -100

        return score

    def iterative_deepening(self, state):
        best_results = None

        self._max_depth = 0
        while True:
            results = minimax_search(state, self)
            if best_results is None or best_results[0] < results[0]:
                best_results = results
            if self._running_time > self._max_running_time:
                break
            self._max_depth += 1

        return best_results[1]

    def _hashable_state(self, state):
        board = state.board.get_json_board()
        list_board = tuple(tuple(l) for l in board)
        return list_board

    def _already_tracked(self, state):
        hashable = self._hashable_state(state)
        return hashable in self._tracking_list

    def _track_state(self, state):
        hashable = self._hashable_state(state)
        self._tracking_list[hashable] += 1

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

    val, action = max_value(state, -inf, inf, 0)
    return val, action
