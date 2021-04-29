#! /usr/bin/env python3
"""NAMES OF THE AUTHOR(S): Gaël Aglin <gael.aglin@uclouvain.be>
                           Vincent Buccilli <vincent.buccilli@student.uclouvain.be>
                           Romain Graux <romain.graux@student.uclouvain.be>"""
import sys
import numpy as np
from search import *
from copy import deepcopy
from itertools import combinations, chain


class BinPacking(Problem):
    def _swap_items(self, state):
        """_swap_items.
        
        yield the next successors when swapping 2 items

        :param state: the current state from which we cant the successors
        """
        # for (key1, val1), (key2, val2) in combinations(state.items.items(), 2):
        for (key1, val1), (key2, val2) in [
            (a, b) for a in state.items.items() for b in state.items.items()
        ]:
            b1, idx1 = state.in_which_bin(key1)
            b2, idx2 = state.in_which_bin(key2)

            # If :
            #   - try to swap in the same bin;
            #   - or both swap items can not fit in their new bin
            # we continue with successors
            if (
                idx1 == idx2
                or not state.can_fit(b1, val2 - val1)
                or not state.can_fit(b2, val1 - val2)
            ):
                continue

            next_bins = deepcopy(state.bins)

            # Delete previous items in the bins
            del next_bins[idx1][key1]
            del next_bins[idx2][key2]

            # Add the swap items
            next_bins[idx1][key2] = val2
            next_bins[idx2][key1] = val1

            yield "swap_items", State(state.capacity, state.items, next_bins)

    def _swap_single(self, state):
        """_swap_single.
        
        yield the next successors when swapping 1 item with possibility to let a bin void

        :param state: the current state from which we cant the successors
        """
        for (key, value), bin_idx, b in [
            (i, idx, b) for i in state.items.items() for idx, b in enumerate(state.bins)
        ]:
            _, item_bin_idx = state.in_which_bin(key)
            # If :
            #   - try to swap within the same bin
            #   - or the item can not fit in the new bin
            # we continue with successors
            if item_bin_idx == bin_idx or not state.can_fit(b, value):
                continue

            next_bins = deepcopy(state.bins)

            # Delete item in previous bin and add it in the new one
            del next_bins[item_bin_idx][key]
            next_bins[bin_idx][key] = value

            # We try to remove void bins
            try:
                next_bins.remove({})
            except ValueError:
                pass

            yield "swap_single", State(state.capacity, state.items, next_bins)

    def successor(self, state):
        for x in chain(self._swap_items(state), self._swap_single(state)):
            yield x

    def fitness(self, state):
        """
        :param state:
        :return: fitness value of the state in parameter
        """
        fitness = 1 - sum(
            [(state.fullness(b.values()) / state.capacity) ** 2 for b in state.bins]
        ) / len(state.bins)
        return -fitness  # Minus in order to maximize

    def value(self, state):
        return self.fitness(state)


class State:
    def __init__(self, capacity, items, bins=None):
        self.capacity = capacity
        self.items = items
        self.bins = bins or self.build_init()

    # an init state building is provided here but you can change it at will
    def build_init(self):
        init = list()
        for ind, size in self.items.items():
            if len(init) == 0 or not self.can_fit(init[-1], size):
                init.append({ind: size})
            else:
                if self.can_fit(init[-1], size):
                    init[-1][ind] = size
        return init

    def can_fit(self, bin, itemsize):
        return sum(list(bin.values())) + itemsize <= self.capacity

    def fullness(self, b):
        return sum(list(b))

    def in_which_bin(self, key):
        for idx, b in enumerate(self.bins):
            if key in b:
                return b, idx
        return None, None

    def __str__(self):
        s = ""
        for i in range(len(self.bins)):
            s += " ".join(list(self.bins[i].keys())) + "\n"
        return s


def read_instance(instanceFile):
    file = open(instanceFile)
    capacitiy = int(file.readline().split(" ")[-1])
    items = {}
    line = file.readline()
    while line:
        items[line.split(" ")[0]] = int(line.split(" ")[1])
        line = file.readline()
    return capacitiy, items


# Attention : Depending of the objective function you use, your goal can be to maximize or to minimize it
def maxvalue(problem, limit=100, callback=None):
    current = LSNode(problem, problem.initial, 0)
    best = current
    best_value = problem.value(current.state)

    for step in range(limit):
        neighbours = list(current.expand())
        values = list(map(lambda n: problem.value(n.state), neighbours))
        best_idx = np.argmax(values)

        current = neighbours[best_idx]

        if values[best_idx] > best_value:
            best = LSNode(problem, current.state, step + 1)
            best_value = values[best_idx]

    return best


# Attention : Depending of the objective function you use, your goal can be to maximize or to minimize it
def randomized_maxvalue(problem, limit=100, n=5, callback=None):
    current = LSNode(problem, problem.initial, 0)
    best = current
    best_value = problem.value(current.state)

    for step in range(limit):
        neighbours = list(current.expand())
        values = list(map(lambda n: problem.value(n.state), neighbours))
        top_idx = np.argpartition(values, -n)[-n:]
        best_idx = np.random.choice(top_idx)

        current = neighbours[best_idx]

        if values[best_idx] > best_value:
            best = LSNode(problem, current.state, step + 1)
            best_value = values[best_idx]

    return best


#####################
#       Launch      #
#####################
if __name__ == "__main__":
    info = read_instance(sys.argv[1])
    init_state = State(info[0], info[1])
    bp_problem = BinPacking(init_state)
    step_limit = 100
    node = randomized_maxvalue(bp_problem, step_limit)
    state = node.state
    print(state)
