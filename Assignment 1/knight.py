# -*-coding: utf-8 -*
'''NAMES OF THE AUTHOR(S): Gael Aglin <gael.aglin@uclouvain.be>
                           Vincent Buccilli <vincent.buccilli@student.uclouvain.be>
                           Romain Graux <romain.graux@student.uclouvain.be>'''

import time
import sys
import copy as copylib
from search import *

# KNIGHT = '♞'
KNIGHT = '♘'

#################
# Problem class #
#################
class Knight(Problem):

    AVAILABLE_MOVES = [(-2,-1), (-1,-2), (2,-1), (-1,2), (2,1), (1,2), (-2,1), (1,-2)]

    def successor(self, state):

        def _border(position):
            # TODO : docstring
            return min(
                    position[0]**2 + position[1]**2, 
                    position[0]**2 + (state.nCols - position[1] - 1)**2, 
                    (state.nRows - position[0] - 1)**2 + position[1]**2, 
                    (state.nCols - position[1] - 1)**2 + (state.nRows - position[0] - 1)**2,
                    )

        def _valid_pos(x, y):
            # TODO : docstring
            return (0 <= x < state.nCols and 0 <= y < state.nRows and state.grid[y][x] != KNIGHT)

        positions = []
        for mov_y, mov_x in Knight.AVAILABLE_MOVES:
            new_x, new_y = state.x+mov_x, state.y+mov_y
            if _valid_pos(new_x, new_y):
                positions.append((new_x, new_y))

        positions = sorted(positions, key = _border, reverse=True)
        for pos in positions:
            new_state = State((state.nRows, state.nCols), pos)
            new_state.n_visited = state.n_visited + 1
            new_state.grid = state.grid
            new_state.grid[pos[0]][pos[1]] = KNIGHT
            yield (0, new_state)

    def goal_test(self, state):
        return state.n_visited == state.nRows * state.nCols


###############
# State class #
###############

class State:
    def __init__(self, shape, init_pos):
        self.nRows, self.nCols = shape
        self.n_visited = 1
        self.grid = []
        for i in range(self.nRows):
            self.grid.append([" "]*self.nCols)
        self.new_state(init_pos, ret=False, copy=False)

    def new_state(self, pos, ret=True, copy=True):
        st = copylib.copy(self) if copy else self
        st.y, st.x = pos
        try:
            st.grid[st.y][st.x] = KNIGHT
        except Exception:
            print(f"x : {st.x} || y : {st.y}")
            print(f"nCols : {st.nCols} || nRows : {st.nRows}")
            print()
        st.n_visited += 1
        if ret:
            return st

    def __str__(self):
        n_sharp = 2 * self.nCols + 1
        s = ("#" * n_sharp) + "\n"
        for i in range(self.nRows):
            s += "#"
            for j in range(self.nCols):
                s = s + str(self.grid[i][j]) + " "
            s = s[:-1]
            s += "#"
            if i < self.nRows - 1:
                s += '\n'
        s += "\n"
        s += "#" * n_sharp
        return s


INGINIOUS = False

if not INGINIOUS:
    ##############################
    # Launch the search in local #
    ##############################
    #Use this block to test your code in local
    # Comment it and uncomment the next one if you want to submit your code on INGInious
    with open('instances.txt') as f:
        instances = f.read().splitlines()
    
    for instance in instances:
        elts = instance.split(" ")
        shape = (int(elts[0]), int(elts[1]))
        init_pos = (int(elts[2]), int(elts[3]))
        init_state = State(shape, init_pos)
    
        problem = Knight(init_state)
    
        # example of bfs tree search
        startTime = time.perf_counter()
        node, nb_explored, remaining_nodes = breadth_first_graph_search(problem)
        endTime = time.perf_counter()
    
        # example of print
        path = node.path()
        path.reverse()
    
        print('Number of moves: ' + str(node.depth))
        for n in path:
            print(n.state)  # assuming that the __str__ function of state outputs the correct format
            print()
        print("* Execution time:\t", str(endTime - startTime))
        print("* Path cost to goal:\t", node.depth, "moves")
        print("* #Nodes explored:\t", nb_explored)
        print("* Queue size at goal:\t",  remaining_nodes)
else:        
    ####################################
    # Launch the search for INGInious  #
    ####################################
    #Use this block to test your code on INGInious
    shape = (int(sys.argv[1]),int(sys.argv[2]))
    init_pos = (int(sys.argv[3]),int(sys.argv[4]))
    init_state = State(shape, init_pos)
    
    problem = Knight(init_state)
    
    # example of bfs tree search
    startTime = time.perf_counter()
    node, nb_explored, remaining_nodes = breadth_first_graph_search(problem)
    endTime = time.perf_counter()
    
    # example of print
    path = node.path()
    path.reverse()
    
    print('Number of moves: ' + str(node.depth))
    for n in path:
        print(n.state)  # assuming that the __str__ function of state outputs the correct format
        print()
    print("* Execution time:\t", str(endTime - startTime))
    print("* Path cost to goal:\t", node.depth, "moves")
    print("* #Nodes explored:\t", nb_explored)
    print("* Queue size at goal:\t",  remaining_nodes)
