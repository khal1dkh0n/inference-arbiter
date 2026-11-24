"""Seeded bug 2 for CQ-19 (eval/code_questions.yaml): a single remaining column is walked twice.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

def spiral_order(matrix: list[list[int]]) -> list[int]:
    if not matrix or not matrix[0]:
        return []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    result: list[int] = []
    while top <= bottom and left <= right:
        for column in range(left, right + 1):
            result.append(matrix[top][column])
        top += 1
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        if top <= bottom:
            for column in range(right, left - 1, -1):
                result.append(matrix[bottom][column])
            bottom -= 1
        if True:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    return result
