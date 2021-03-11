#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mar 10, 09:20:31
@last modified : 2021 Mar 11, 16:24:58
"""

"""NAMES OF THE AUTHOR(S): Gael Aglin <gael.aglin@uclouvain.be>
                           Vincent Buccilli <vincent.buccilli@student.uclouvain.be>
                           Romain Graux <romain.graux@student.uclouvain.be>"""
# TODO: Prune dead states
# TODO: implement heuristic class
# TODO: change hash with ASCII

INGINIOUS = True
MYTEST = False
BENCHMARK = False

from search import *
import sys
import time
import numpy as np
from collections import defaultdict

goal_state = None
#################
# Problem class #
#################
class Blocks(Problem):
    WALL = "#"
    VOID = " "
    FIXED = "@"
    AVAILABLE_MOVES = [(0, -1), (0, 1)]  # i.e. LEFT, RIGHT

    def __init__(self, init_state, goal_state=None):
        State.goal = goal_state
        super(Blocks, self).__init__(init_state, goal_state)

    @staticmethod
    def borders(state, pos):
        """borders.
        return True if the positon `pos` is between the borders else False

        :param state: the state containing the blocks, goal and walls
        :param pos: (y, x)
        """
        y, x = pos
        return 0 <= y < state.nbr and 0 <= x < state.nbc

    @staticmethod
    def valid_position(state, pos):
        """valid_position.
        return True if the position `pos` is between the borders and free else
        False

        :param state: the state containing the blocks, goal and walls
        :param pos: (y, x)
        """
        y, x = pos
        return Blocks.borders(state, pos) and state[y, x] == Blocks.VOID

    @staticmethod
    def have_support(state, pos):
        """have_support.
        return True if the block at position `pos` is supported by a WALL, a
        border or another block.

        :param state:
        :param pos:
        """
        y, x = pos
        return y == (state.nbr - 1) or state[y + 1, x] != Blocks.VOID

    @staticmethod
    def move_and_apply_gravity(state, pos, prev_pos):
        """move_and_apply_gravity.
        apply the gravity for a position and block above the previous position

        :param state:
        :param pos:
        :param prev_pos: y, x of the previous position
        """

        def inner_apply_gravity(state, pos, prev_pos=None):
            iy, _ = y, x = pos
            while not Blocks.have_support(state, (y, x)):
                y += 1
            return state.new_state((y, x), prev_pos or pos)

        next_state = inner_apply_gravity(state, pos, prev_pos)

        iy, ix = prev_pos
        blocks = 1
        while Blocks.borders(next_state, (iy - blocks, ix)) and next_state.blocks.get(
            (iy - blocks, ix), "VOID"
        ) not in (Blocks.FIXED, "VOID"):
            next_state = inner_apply_gravity(next_state, (iy - blocks, ix))
            blocks += 1

        return next_state

    def movable_blocks(self, state):
        """movable_blocks.
        return all movable blocks (not at goal position)

        :param state:
        """
        for (y, x), cls in state.blocks.items():
            if cls != Blocks.FIXED:
                yield y, x

    @staticmethod
    def is_dead_state(state):
        """is_dead_state.
        return if the state is a dead state or not

        :param state:
        """

        def lower_than_goal():
            """lower_than_goal.
            check if at least one block is above the highest goal block
            """

            def naive():
                blocks_y = [y for y, x in state.blocks.keys()]
                goals_y = [y for y, x in state.goal.blocks.keys()]
                if min(blocks_y) > min(goals_y):
                    return True
                return False

            return naive()

        def possible_positions():
            """possible_positions.
            check if the goal blocks are still accessible by the corresponding
            blocks
            """

            def bfs(init):
                """bfs.
                return the position visited by the bfs

                :param init: the initial position (goal position) y, x
                """

                def get_neighbors(pos):
                    """get_neighbors.
                    return the neighbors of the position `pos` (left, right and
                    upper direction)

                    :param pos:
                    """
                    y, x = pos
                    moves = [(0, -1), (0, 1), (-1, 0)]
                    neighbors = []
                    for dy, dx in moves:
                        yy, xx = y + dy, x + dx
                        if (
                            0 <= yy < state.nbr
                            and 0 <= xx < state.nbc
                            and state[(yy, xx)] not in (Blocks.WALL, Blocks.FIXED)
                        ):
                            neighbors.append((yy, xx))
                    return neighbors

                visited = set()
                queue = [init]

                while queue:
                    node = queue.pop(0)
                    if node not in visited:
                        visited.add(node)
                        neighbors = get_neighbors(node)
                        for neighbor in neighbors:
                            queue.append(neighbor)
                return visited

            # compute the accessible positions for each goal block
            goal_accessible = defaultdict(list)
            for pos, cls in state.goal.blocks.items():
                if not state.is_at_goal(pos):
                    goal_accessible[cls.lower()].append(bfs(pos))

            # for each block, check if they are accessible by the goal block of
            # the same class (for each goal block)
            set_cls = defaultdict(list)
            for pos, cls in state.blocks.items():
                if state.is_at_goal(pos):
                    continue
                n_poss = 0b0
                for idx, (goal_cls, goal_maps) in enumerate(goal_accessible.items()):
                    for goal_map in goal_maps:
                        n_poss += (
                            1 << idx
                            if (cls == goal_cls and pos in goal_map)
                            else 0  # Encode 1 bit per goal block if accessible
                        )
                set_cls[cls].append(n_poss)

            for cls, list_poss in set_cls.items():
                len_goals = len(goal_accessible[cls])

                # No goal for this class
                if len_goals == 0:
                    continue

                b = 0b0
                for bits in list_poss:
                    b = b | bits
                if b < (
                    2 ** (len_goals - 1)
                ):  # check if at least one goal position can not be accessed by a block
                    return True

            return False

        return lower_than_goal() or possible_positions()

    def successor(self, state):
        """successor.
        return the successor (i.e. all possible moves for each block instance)

        drop :
            - blocks already at goal position
            - dead states
        - try all possible moves :: check validity + apply gravity

        :param state:
        """
        for y, x in self.movable_blocks(state):
            for dy, dx in Blocks.AVAILABLE_MOVES:
                new_pos = y + dy, x + dx
                if Blocks.valid_position(state, new_pos):
                    next_state = Blocks.move_and_apply_gravity(state, new_pos, (y, x))
                    if not Blocks.is_dead_state(next_state):
                        yield 0, next_state

    def goal_test(self, state):
        for (y, x), cls in self.goal.blocks.items():
            if not state.is_at_goal((y, x)):
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

            # Encode the blocks as a hash map :: position -> the block class(a, b, ...)
            self.blocks = dict(
                zip(
                    [(y, x) for y, x in blocks_idx],
                    [grid[y, x] for y, x in blocks_idx],
                )
            )

            # Encode the walls as hash map :: position -> WALL (#)
            State.walls = dict(
                zip([(y, x) for y, x in walls_idx], [Blocks.WALL] * len(walls_idx))
            )

            del grid
            del blocks_idx
            del walls_idx

    def new_state(self, new_pos, prev_pos=None):
        y, x = new_pos

        # Update the new position
        new_blocks = self.blocks.copy()
        new_blocks[new_pos] = self[prev_pos]
        del new_blocks[prev_pos]

        new_state = State(grid=new_blocks)
        # Set as fixed block if is at goal
        if new_state.is_at_goal(new_pos):
            new_state.blocks[new_pos] = Blocks.FIXED

        return new_state

    def is_at_goal(self, position):
        cls = self.blocks.get(position)

        if cls == Blocks.FIXED:
            return True

        cls_goal = self.goal.blocks.get(position)
        if cls_goal is not None:
            if cls_goal.lower() == cls:
                return True
        return False

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
        return hash(str(self))
        n_bits = 8  # ASCII encoding
        h = 0
        for (y, x), cls in self.blocks.items():
            h += (self.nbr * y + x) * n_bits + ord(cls)
        return int(h)


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


class Heuristic:
    @staticmethod
    def at_good_position(node):
        h = 0.0
        state = node.state
        for cls in state.blocks.values():
            if cls == Blocks.FIXED:
                h += 1
        return h

    @staticmethod
    def how_many_fill_block(node):
        h = 0.0
        state = node.state
        for (y, x), cls in state.goal.blocks.items():
            if not state.is_at_goal((y, x)):
                blocks = 1
                while not Blocks.have_support(state, (y + blocks, x)):
                    h += 1
                    blocks += 1
        return h

    @staticmethod
    def manhattan(node):
        def distance(pos1, pos2):
            return np.abs(pos1[1] - pos2[1])

        state = node.state
        goal = state.goal

        # Change the mapping to cls of the block -> a list of all positions in the grid
        cls_to_pos = defaultdict(list)
        for pos, cls in state.blocks.items():
            cls_to_pos[cls].append(pos)

        def closest(goal_position, goal_cls):
            """closest.

            :param goal_position: y, x of the goal position
            :param goal_cls: the block class of the goal
            """
            if state.is_at_goal(goal_position):
                return 0
            list_pos = cls_to_pos.get(goal_cls.lower())
            if list_pos is not None:
                all_pos = list(map(lambda pos: distance(goal_position, pos), list_pos))
                return min(all_pos)
            return 0

        all_closest_distances = [
            closest(pos, cls) for pos, cls in goal.blocks.items()
        ]  # Iterate over all goal positions (return the closest distance for each goal)
        return max(all_closest_distances, default=0)


def heuristic(node):
    # hs = list(
    #     map(
    #         lambda f: f(node),
    #         [Heuristic.at_good_position, Heuristic.how_many_fill_block],
    #     )
    # )
    # return np.max(hs)
    return Heuristic.manhattan(node)


##############################
# Launch the search in local #
##############################
# Use this block to test your code in local
# Comment it and uncomment the next one if you want to submit your code on INGInious

if __name__ == "__main__":

    if not INGINIOUS and not MYTEST and not BENCHMARK:
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
    elif INGINIOUS and not MYTEST and not BENCHMARK:

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
        # node, nb_explored, remaining_nodes = astar_graph_search(problem, heuristic)
        node, nb_explored, remaining_nodes = breadth_first_graph_search(problem)
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
    elif BENCHMARK:
        instance = sys.argv[1]
        grid_init, grid_goal = readInstanceFile(instance)
        init_state = State(grid_init)
        goal_state = State(grid_goal)
        problem = Blocks(init_state, goal_state)

        # example of bfs graph search
        startTime = time.perf_counter()
        node, nb_explored, remaining_nodes = astar_graph_search(problem, heuristic)
        # node, nb_explored, remaining_nodes = breadth_first_graph_search(problem)
        endTime = time.perf_counter()

        # example of print
        path = node.path()
        path.reverse()

        args = [
            str(endTime - startTime),
            str(node.depth),
            str(nb_explored),
            str(remaining_nodes),
        ]
        s = ",".join(args)
        print(s)

    else:
        instances_path = "instances/"
        instance_names = [
            # "mine",
            "a01",
            # "a02",
            # "a03",
            # "a04",
            # "a05",
            # "a06",
            # "a07",
            # "a08",
            # "a09",
            # "a10",
        ]

        class Node:
            def __init__(self, state):
                self.state = state

        for instance in [instances_path + name for name in instance_names]:
            grid_init, grid_goal = readInstanceFile(instance)
            init_state = State(grid_init)
            goal_state = State(grid_goal)
            problem = Blocks(init_state, goal_state)
            print(init_state, "\n" * 2)
            for idx, (_, new_state) in enumerate(problem.successor(init_state)):
                print("=" * 50)
                print(new_state)
                print(heuristic(Node(new_state)))
