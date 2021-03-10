#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mar 10, 09:20:31
@last modified : 2021 Mar 10, 10:47:36
"""

"""NAMES OF THE AUTHOR(S): Gael Aglin <gael.aglin@uclouvain.be>
                           Vincent Buccilli <vincent.buccilli@student.uclouvain.be>
                           Romain Graux <romain.graux@student.uclouvain.be>"""

INGINIOUS = False
MYTEST = True

import numpy as np
from search import *
import sys
import time

goal_state = None
#################
# Problem class #
#################
class Blocks(Problem):
    WALL = "#"
    VOID = " "
    AVAILABLE_MOVES = [(0, -1), (0, 1)]  # i.e. LEFT, RIGHT

    @staticmethod
    def valid_position(state, pos):
        y, x = pos
        return (
            0 <= y < state.nbr
            and 0 <= x < state.nbc
            and state.grid[y, x] == Blocks.VOID
        )

    @staticmethod
    def apply_gravity(state, pos):
        def have_support(state, pos):
            y, x = pos
            return y == (state.nbr - 1) or state.grid[y + 1, x] == Blocks.WALL

        y, x = pos
        while not have_support(state, (y, x)):
            y += 1
        return (y, x)

    def successor(self, state):
        """successor.
        return the successor (i.e. all possible moves for each block instance)
        - identify position of all blocks (return only blocks not touching a
          goal position)
        - try all possible moves :: check validity + apply gravity

        :param state:
        """
        for block_id, (y, x) in enumerate(state.blocks):
            for dy, dx in Blocks.AVAILABLE_MOVES:
                new_pos = y + dy, x + dx
                if Blocks.valid_position(state, new_pos):
                    new_pos = Blocks.apply_gravity(state, new_pos)
                    yield 0, state.new_state(block_id, new_pos)

    def goal_test(self, state):
        pass


###############
# State class #
###############
class State:
    def __init__(self, grid):
        self.nbr = len(grid)
        self.nbc = len(grid[0])
        self.grid = np.array(grid)
        self.blocks = np.array(
            np.argwhere(
                np.logical_and(self.grid != Blocks.WALL, self.grid != Blocks.VOID)
            )
        )

    def new_state(self, block_id, new_pos):
        y, x = new_pos
        prev_y, prev_x = self.blocks[block_id]
        cls = self.grid[prev_y, prev_x]
        new_grid = self.grid.copy()
        # if new_grid[y, x] != Blocks.VOID:
        #     raise Exception(f"Not void position for {y, x} :: {new_grid[y, x]}")
        new_grid[prev_y, prev_x] = Blocks.VOID
        new_grid[y, x] = cls
        return State(new_grid)

    def __str__(self):
        n_sharp = self.nbc + 2
        s = ("#" * n_sharp) + "\n"
        for i in range(self.nbr):
            s += "#"
            for j in range(self.nbc):
                s = s + str(self.grid[i][j])
            s += "#"
            if i < self.nbr - 1:
                s += "\n"
        return s + "\n" + "#" * n_sharp


######################
# Auxiliary function #
######################
def readInstanceFile(filename):
    grid_init, grid_goal = map(
        lambda x: [[c for c in l.rstrip("\n")[1:-1]] for l in open(filename + x)],
        [".init", ".goalinfo"],
    )
    return grid_init[1:-1], grid_goal[1:-1]


######################
# Heuristic function #
######################
def heuristic(node):
    h = 0.0
    # ...
    # compute an heuristic value
    # ...
    return h


##############################
# Launch the search in local #
##############################
# Use this block to test your code in local
# Comment it and uncomment the next one if you want to submit your code on INGInious

if __name__ == "__main__":

    if not INGINIOUS and not MYTEST:
        instances_path = "instances/"
        instance_names = [
            "a01",
            "a02",
            "a03",
            "a04",
            "a05",
            "a06",
            "a07",
            "a08",
            "a09",
            "a10",
            "a11",
        ]

        for instance in [instances_path + name for name in instance_names]:
            grid_init, grid_goal = readInstanceFile(instance)
            init_state = State(grid_init)
            goal_state = State(grid_goal)
            problem = Blocks(init_state)

            # example of bfs tree search
            startTime = time.perf_counter()
            node, nb_explored, remaining_nodes = depth_first_graph_search(problem)
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
    elif INGINIOUS and not MYTEST:

        ####################################
        # Launch the search for INGInious  #
        ####################################
        # Use this block to test your code on INGInious
        instance = sys.argv[1]
        grid_init, grid_goal = readInstanceFile(instance)
        init_state = State(grid_init)
        goal_state = State(grid_goal)
        problem = Blocks(init_state)

        # example of bfs graph search
        startTime = time.perf_counter()
        node, nb_explored, remaining_nodes = astar_graph_search(problem, heuristic)
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
        instances_path = "instances/"
        instance_names = [
            "mine",
            # "a01",
            # "a02",
            # "a03",
            # "a04",
            # "a05",
            # "a06",
            # "a07",
            # "a08",
            # "a09",
            # "a10",
            # "a11",
        ]

        for instance in [instances_path + name for name in instance_names]:
            grid_init, grid_goal = readInstanceFile(instance)
            init_state = State(grid_init)
            problem = Blocks(init_state)
            print(init_state, "\n" * 2)
            for _, new_state in problem.successor(init_state):
                print(new_state)
