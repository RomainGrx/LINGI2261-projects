#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 30, 11:31:30
@last modified : 2021 Apr 30, 14:33:51
"""


def get_logic_table(expression):
    import inspect
    from itertools import product

    results = dict()
    args = inspect.getargs(expression.__code__).args

    for possibility in product(range(2), repeat=len(args)):
        value = expression(*possibility)
        results[possibility] = value

    return results


def get_number_valid_propositions(expression):
    values = list(get_logic_table(expression).values())
    return sum(values)


implies = lambda x, y: not x or y

s0 = lambda a, b, c, d: (not a or c) and (not b or c)
s1 = lambda a, b, c, d: implies(c, not a) and not (b or c)
s2 = lambda a, b, c, d: (not a or b) and not implies(b, not c) and not implies(not d, a)

from inspect import getsource

for s in (s0, s1, s2):
    valid = get_number_valid_propositions(s)
    print(f'{getsource(s).split("=")[0]} :: {valid}')
