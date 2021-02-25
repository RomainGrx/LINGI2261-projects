#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Feb 13, 12:10:58
@last modified : 2021 Feb 25, 11:03:18
"""
"""NAMES OF THE AUTHOR(S): Gael Aglin <gael.aglin@uclouvain.be>
                           Vincent Buccilli <vincent.buccilli@student.uclouvain.be>
                           Romain Graux <romain.graux@student.uclouvain.be>"""

import time
import sys
import copy as copylib
from functools import partial
from search import *

KNIGHT = "♘"  # u"\u2658"
VISITED_TILE = "♞"  # u"\u265E"
NOT_VISITED_TILE = " "

#################
#   Key class   #
#################


class Key:
    """Key is a class containing different algorithm used to sort states. The
    algorithms try to return highest value for more promising positions so we
    can sort these positions and return the best one first in Knight.successor.
    """

    @classmethod
    def naive(cls, position, state):
        """Return 1 whatever the position

        :param position: possible next position (y, x)
        :param state: current state
        """
        return 1

    @classmethod
    def border(cls, position, state):
        """Give the distance between the current position and the closest
        border

        :param position: possible next position (y, x)
        :param state: current state
        """
        y, x = position
        return min(
            y ** 3 + x ** 2,
            y ** 3 + (state.nCols - x - 1) ** 2,
            (state.nRows - y - 2) ** 2 + x ** 2,
            (state.nCols - x - 2) ** 2 + (state.nRows - y - 1) ** 2,
        )

    @classmethod
    def neighbors(cls, position, state):
        """Give the number of neighbors tiles not already visited

        :param position: possible next position (y, x)
        :param state: current state
        """
        y, x = position
        count = 1
        for dy, dx in Knight.AVAILABLE_MOVES:
            yy, xx = y + dy, x + dx
            if Knight.valid_pos(yy, xx, state):
                count += 2
        return count


#################
# Problem class #
#################
class Knight(Problem):
    """Knight."""

    AVAILABLE_MOVES = [
        (-2, -1),
        (-1, -2),
        (2, -1),
        (-1, 2),
        (2, 1),
        (1, 2),
        (-2, 1),
        (1, -2),
    ]  # List all possible moves for a knight (L shape)

    @classmethod
    def valid_pos(cls, y, x, state):
        """Return True if the position is within the borders and
        the position is not yet visited, else False

        :param y: y coordinate of the position
        :param x: x coordinate of the position
        """
        return (
            0 <= x < state.nCols
            and 0 <= y < state.nRows
            and state.grid[y][x] != VISITED_TILE
        )

    def successor(self, state):
        """Yield all possible next states in descending order (order given by `Key.*`) so the search can find the goal state as fast as possible.

        :param state: the current state
        """
        # Append all valid positions to the `positions` list
        positions = []
        for dy, dx in Knight.AVAILABLE_MOVES:
            new_pos = state.y + dy, state.x + dx
            if Knight.valid_pos(*new_pos, state):
                positions.append(new_pos)

        # Sort positions in descending order with order given by Key.*
        positions = sorted(
            positions, key=partial(Key.neighbors, state=state), reverse=True
        )
        for pos in positions:
            new_state = state.next_state(pos)
            yield (0, new_state)

    def goal_test(self, state):
        """Return True if the state is the goal state i.e. we have visited all
        tiles (nRows * nCols)

        :param state: the current state
        """
        return state.n_visited == state.nRows * state.nCols


###############
# State class #
###############


class State:
    """State."""

    def __init__(self, shape, init_pos, grid=None, n_visited=1):
        """__init__.

        :param shape:
        :param init_pos:
        """
        self.shape = self.nRows, self.nCols = shape
        self.n_visited = n_visited
        if grid is None:
            self.grid = [[" " for i in range(self.nCols)] for j in range(self.nRows)]
        else:
            self.grid = [[v for v in r] for r in grid]

        self.init_pos = self.y, self.x = init_pos
        self.grid[self.y][self.x] = KNIGHT

    def next_state(self, position):
        """Return the next state corresponding to the same state as self with
        new position

        :param position: the new position (y, x)
        """
        prev_y, prev_x = self.y, self.x
        st = State(self.shape, position, grid=self.grid, n_visited=self.n_visited + 1)
        st.grid[prev_y][prev_x] = VISITED_TILE
        return st

    def __str__(self):
        """__str__."""
        n_sharp = 2 * self.nCols + 1
        s = ("#" * n_sharp) + "\n"
        for i in range(self.nRows):
            s += "#"
            for j in range(self.nCols):
                s = s + str(self.grid[i][j]) + " "
            s = s[:-1]
            s += "#"
            if i < self.nRows - 1:
                s += "\n"
        s += "\n"
        s += "#" * n_sharp
        return s

    def __eq__(self, other):
        """__eq__.

        :param other: the other state needed to compare with self
        """
        if (
            (self.n_visited != other.n_visited)
            or (self.x != other.x)
            or (self.y != other.y)
        ):
            return False
        for i in range(self.nRows):
            for j in range(self.nCols):
                if self.grid[i][j] != other.grid[i][j]:
                    return False
        return True

    def __hash__(self):
        """__hash__ returns the hash of the state. The hash is computed such
        that symmetrical states return the same hash"""
        return hash(str(self))


INGINIOUS = True

if __name__ == "__main__":
    if not INGINIOUS:
        ##############################
        # Launch the search in local #
        ##############################
        # Use this block to test your code in local
        # Comment it and uncomment the next one if you want to submit your code on INGInious
        with open("instances.txt") as f:
            instances = f.read().splitlines()

        for instance in instances:
            elts = instance.split(" ")
            shape = (int(elts[0]), int(elts[1]))
            init_pos = (int(elts[2]), int(elts[3]))
            init_state = State(shape, init_pos)

            problem = Knight(init_state)

            # example of bfs tree search
            startTime = time.perf_counter()
            # node, nb_explored, remaining_nodes = breadth_first_graph_search(problem)
            node, nb_explored, remaining_nodes = depth_first_tree_search(problem)
            endTime = time.perf_counter()

            # example of print
            path = node.path()
            path.reverse()

            print("Number of moves: " + str(node.depth))
            for n in path:
                print(
                    n.state
                )  # assuming that the __str__ function of state outputs the correct format
                print()
            print("* Execution time:\t", str(endTime - startTime))
            print("* Path cost to goal:\t", node.depth, "moves")
            print("* #Nodes explored:\t", nb_explored)
            print("* Queue size at goal:\t", remaining_nodes)
    else:
        ####################################
        # Launch the search for INGInious  #
        ####################################
        # Use this block to test your code on INGInious
        shape = (int(sys.argv[1]), int(sys.argv[2]))
        init_pos = (int(sys.argv[3]), int(sys.argv[4]))
        init_state = State(shape, init_pos)

        problem = Knight(init_state)

        # example of bfs tree search
        startTime = time.perf_counter()
        # node, nb_explored, remaining_nodes = breadth_first_graph_search(problem)
        node, nb_explored, remaining_nodes = depth_first_tree_search(problem)
        endTime = time.perf_counter()

        # example of print
        path = node.path()
        path.reverse()

        print("Number of moves: " + str(node.depth))
        for n in path:
            print(
                n.state
            )  # assuming that the __str__ function of state outputs the correct format
            print()
        print("* Execution time:\t", str(endTime - startTime))
        print("* Path cost to goal:\t", node.depth, "moves")
        print("* #Nodes explored:\t", nb_explored)
        print("* Queue size at goal:\t", remaining_nodes)
