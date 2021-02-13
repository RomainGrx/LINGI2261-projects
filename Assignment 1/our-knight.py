#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Feb 12, 18:01:39
@last modified : 2021 Feb 13, 11:13:43
"""

import argparse
from utils import get_knight_args

def knight_solver(n_rows, n_columns, x_start, y_start):
    raise NotImplementedError()

if __name__=="__main__":
    args = get_knight_args()

    knight_solver(args.n_rows, args.n_columns, args.x_start, args.y_start)

