#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Feb 20, 10:19:00
@last modified : 2021 Feb 20, 10:31:07
"""

import unittest
from knight import Key, Knight, State, VISITED_TILE, NOT_VISITED_TILE, KNIGHT

class StateTest(unittest.TestCase):
    def test_init(self):
        shape = (5, 5)
        pos = y, x = (2, 2)
        state = State(shape, pos)
        self.assertEqual(state.n_visited, 1)
        self.assertEqual(state.nRows, shape[0])
        self.assertEqual(state.nCols, shape[1])
        self.assertEqual(state.grid[y][x], KNIGHT)
        for yy, r in enumerate(state.grid):
            for xx, v in enumerate(r):
                if yy == y and xx == x : continue
                self.assertEqual(v, NOT_VISITED_TILE)

    def test_next_state(self):
        shape = (5, 5)
        init_pos = y, x = (2, 2)
        init_state = State(shape, init_pos)
        dy, dx = -1, -2 
        next_pos = yy, xx = y+dy, x+dx
        next_state = init_state.next_state(next_pos)
        self.assertEqual(next_state.n_visited, 2)
        self.assertEqual(next_state.grid[y][x], VISITED_TILE)
        self.assertEqual(next_state.grid[yy][xx], KNIGHT)
        for yyy, r in enumerate(next_state.grid):
            for xxx, v in enumerate(r):
                if yyy == y and xxx == x or yyy == yy and xxx == xx : continue
                self.assertEqual(v, NOT_VISITED_TILE)
            

class KnightTest(unittest.TestCase):
    def test_init(self):
        pass

if __name__ == '__main__':
    unittest.main()
