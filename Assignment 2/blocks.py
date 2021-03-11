#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mar 10, 09:20:31
@last modified : 2021 Mar 11, 01:29:48
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
    def borders(state, pos):
        y, x = pos
        return 0 <= y < state.nbr and 0 <= x < state.nbc

    @staticmethod
    def valid_position(state, pos):
        y, x = pos
        return Blocks.borders(state, pos) and state[y, x] == Blocks.VOID

    @staticmethod
    def move_and_apply_gravity(state, pos, prev_pos):
        def have_support(state, pos):
            y, x = pos
            return y == (state.nbr - 1) or state[y + 1, x] != Blocks.VOID

        def inner_apply_gravity(state, pos, prev_pos=None):
            iy, _ = y, x = pos
            while not have_support(state, (y, x)):
                y += 1
            return state.new_state((y, x), prev_pos or pos) 

        next_state = inner_apply_gravity(state, pos, prev_pos)

        iy, ix = prev_pos
        blocks = 1
        while Blocks.borders(next_state, (iy-blocks, ix)) and next_state.blocks.get((iy-blocks, ix), "VOID") != "VOID":
            next_state = inner_apply_gravity(next_state, (iy-blocks, ix))
            blocks += 1
        
        return next_state

    def movable_blocks(self, state):
        for (y, x), cls in state.blocks.items():
            if self.goal[y, x] != cls.upper():
                yield y, x

    def successor(self, state):
        """successor.
        return the successor (i.e. all possible moves for each block instance)
        - identify position of all blocks (return only blocks not touching a
          goal position)
        - try all possible moves :: check validity + apply gravity

        :param state:
        """
        if State.goal is None:
            State.goal = self.goal
        for y, x in self.movable_blocks(state):
            for dy, dx in Blocks.AVAILABLE_MOVES:
                new_pos = y + dy, x + dx
                if Blocks.valid_position(state, new_pos):
                    next_state = Blocks.move_and_apply_gravity(state, new_pos, (y, x))
                    yield 0, next_state

    def goal_test(self, state):
        for (y, x), cls in self.goal.blocks.items():
            if state[y, x] != cls.lower():
                return False
        return True


###############
# State class #
###############
class State:
    walls, nbr, nbc, goal = None, None, None, None

    def __init__(self, grid=None, blocks=None):
        assert grid is not None or blocks is not None, "need at least one argument"
        if isinstance(grid, dict):
            self.blocks = grid
        else:
            State.nbr = len(grid)
            State.nbc = len(grid[0])
            grid = np.array(grid)

            blocks_idx = np.array(
                np.argwhere(np.logical_and(grid != Blocks.WALL, grid != Blocks.VOID))
            )
            walls_idx = np.array(np.argwhere(grid == Blocks.WALL))

            self.blocks = dict(
                zip(
                    [(y, x) for y, x in blocks_idx],
                    [grid[y, x] for y, x in blocks_idx],
                )
            )

            State.walls = dict(
                zip([(y, x) for y, x in walls_idx], [Blocks.WALL] * len(walls_idx))
            )

            del grid
            del blocks_idx
            del walls_idx

    def new_state(self, new_pos, prev_pos=None):
        y, x = new_pos

        new_blocks = self.blocks.copy()
        new_blocks[new_pos] = self[prev_pos]
        del new_blocks[prev_pos]

        new_state = State(grid=new_blocks)

        return new_state

    def __getitem__(self, attr):
        return self.blocks.get(attr, None) or State.walls.get(attr, Blocks.VOID)

    def __str__(self):
        n_sharp = self.nbc + 2
        s = ("#" * n_sharp) + "\n"
        for i in range(self.nbr):
            s += "#"
            for j in range(self.nbc):
                s = s + str(self[i, j])
            s += "#"
            if i < self.nbr - 1:
                s += "\n"
        return s + "\n" + "#" * n_sharp

    def __eq__(self, other):
        assert self.nbc == other.nbc and self.nbr == other.nbr
        assert isinstance(other, State)
        return self.blocks == other.blocks

    def __hash__(self):
        unique_blocks = np.unique(list(self.blocks.values()))
        CODEX = dict(zip(unique_blocks, np.arange(len(unique_blocks))))
        n_bits = int(np.ceil(np.log2(len(unique_blocks))))
        hashh = 0
        for (y, x), cls in self.blocks.items():
            hashh += (self.nbr * y + x) * n_bits + CODEX[cls]
        return int(hashh)


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
            problem = Blocks(init_state, goal_state)

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
        problem = Blocks(init_state, goal_state)

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
            goal_state = State(grid_goal)
            problem = Blocks(init_state, goal_state)
            print(init_state, "\n" * 2)
            for idx, (_, new_state) in enumerate(problem.successor(init_state)):
                print(new_state)
