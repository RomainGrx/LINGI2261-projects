#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Feb 13, 11:09:42
@last modified : 2021 Feb 13, 11:11:44
"""

import argparse

def get_knight_args(desc="Knight solver"):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('n_rows', type=int, help="number of rows")
    parser.add_argument('n_columns', type=int, help="number of columns")
    parser.add_argument('y_start', type=int, help="y coordinate of starting point")
    parser.add_argument('x_start', type=int, help="x coordinate of starting point")
    args = parser.parse_args()

    return args

