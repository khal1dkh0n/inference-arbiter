"""Reference tests for CQ-19 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import spiral_order


class SpiralOrderTests(unittest.TestCase):
    def test_square(self):
        self.assertEqual(spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [1, 2, 3, 6, 9, 8, 7, 4, 5])
        self.assertEqual(spiral_order([[1, 2], [3, 4]]), [1, 2, 4, 3])

    def test_wide_rectangle(self):
        matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        self.assertEqual(spiral_order(matrix), [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7])

    def test_tall_rectangle(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        self.assertEqual(spiral_order(matrix), [1, 2, 3, 6, 9, 12, 11, 10, 7, 4, 5, 8])

    def test_single_row_and_single_column(self):
        self.assertEqual(spiral_order([[1, 2, 3]]), [1, 2, 3])
        self.assertEqual(spiral_order([[1], [2], [3]]), [1, 2, 3])

    def test_inner_single_row_and_column(self):
        self.assertEqual(
            spiral_order([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]),
            [1, 2, 3, 4, 5, 10, 15, 14, 13, 12, 11, 6, 7, 8, 9],
        )
        self.assertEqual(
            spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]),
            [1, 2, 3, 6, 9, 12, 15, 14, 13, 10, 7, 4, 5, 8, 11],
        )

    def test_one_element(self):
        self.assertEqual(spiral_order([[7]]), [7])

    def test_empty(self):
        self.assertEqual(spiral_order([]), [])
        self.assertEqual(spiral_order([[]]), [])
        self.assertEqual(spiral_order([[], []]), [])

    def test_matrix_is_not_modified(self):
        matrix = [[1, 2], [3, 4]]
        spiral_order(matrix)
        self.assertEqual(matrix, [[1, 2], [3, 4]])
