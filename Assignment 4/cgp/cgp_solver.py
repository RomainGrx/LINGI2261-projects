from clause import *

"""
For the color grid problem, the only code you have to do is in this file.

You should replace

# your code here

by a code generating a list of clauses modeling the grid color problem
for the input file.

You should build clauses using the Clause class defined in clause.py

Read the comment on top of clause.py to see how this works.
"""

from itertools import product


def get_expression(size, points=None):
    expression = []

    if points is not None:
        for i, j, k in points:
            clause = Clause(size)
            clause.add_positive(i, j, k)
            expression.append(clause)

    def moves(alpha):
        direction = (0, alpha, -alpha)
        possible_moves = list(
            product(direction, direction)
        )  # Will produce (0, alpha), (alpha, 0), (alpha, alpha), (-alpha, alpha), ... (8 moves without (0,0))
        possible_moves.remove((0, 0))  # Remove this move
        return possible_moves

    for i in range(size):
        for j in range(size):
            clause_color = Clause(size)  # A color for each case clause
            for k in range(size):
                clause_color.add_positive(i, j, k)  # Get at least a color per case
                for alpha in range(size):
                    # Get all directions (i.e. within the same row, column or diagonal)
                    for (di, dj) in moves(alpha):
                        x, y = i + di, j + dj

                        # Within the limit of the board
                        if not 0 <= x < size or not 0 <= y < size:
                            continue

                        if x == i and y == j:
                            continue

                        clause = Clause(size)
                        clause.add_negative(i, j, k)
                        clause.add_negative(x, y, k)
                        expression.append(clause)

            expression.append(clause_color)

    return expression


if __name__ == "__main__":
    expression = get_expression(3)
    for clause in expression:
        print(clause)
