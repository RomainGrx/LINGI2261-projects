#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Feb 12, 18:01:39
@last modified : 2021 Feb 12, 18:28:02
"""

import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Knight solver")
    parser.add_argument('n_rows', type=int, help="number of rows")
    parser.add_argument('n_columns', type=int, help="number of columns")
    parser.add_argument('y_start', type=int, help="y coordinate of starting point")
    parser.add_argument('x_start', type=int, help="x coordinate of starting point")
    args = parser.parse_args()


