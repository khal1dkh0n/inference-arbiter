"""Reference tests for CQ-13 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import allocate


class AllocateTests(unittest.TestCase):
    def test_even_split_with_one_unit_left(self):
        self.assertEqual(allocate(100, [1, 1, 1]), [34, 33, 33])

    def test_exact_split(self):
        self.assertEqual(allocate(100, [50, 30, 20]), [50, 30, 20])
        self.assertEqual(allocate(7, [1, 2, 4]), [1, 2, 4])

    def test_largest_remainder_gets_the_unit(self):
        self.assertEqual(allocate(10, [1, 2]), [3, 7])
        self.assertEqual(allocate(10, [2, 1]), [7, 3])
        self.assertEqual(allocate(100, [1, 2, 3]), [17, 33, 50])

    def test_equal_remainders_go_to_the_lower_index(self):
        self.assertEqual(allocate(5, [3, 7]), [2, 3])
        self.assertEqual(allocate(2, [1, 1, 1]), [1, 1, 0])
        self.assertEqual(allocate(1, [1, 1, 1, 1]), [1, 0, 0, 0])

    def test_zero_weight_gets_nothing(self):
        self.assertEqual(allocate(10, [0, 1]), [0, 10])
        self.assertEqual(allocate(10, [1, 0, 1]), [5, 0, 5])

    def test_zero_total(self):
        self.assertEqual(allocate(0, [1, 2]), [0, 0])

    def test_large_totals_are_exact(self):
        self.assertEqual(allocate(10**18 + 1, [1, 1]), [5 * 10**17 + 1, 5 * 10**17])

    def test_shares_add_up(self):
        shares = allocate(1000, [3, 3, 3, 1, 7, 11])
        self.assertEqual(sum(shares), 1000)
        self.assertEqual(shares, [107, 107, 107, 36, 250, 393])

    def test_invalid_arguments_raise(self):
        for total, weights in ((-1, [1]), (10, []), (10, [1, -1]), (10, [0, 0])):
            with self.subTest(total=total, weights=weights):
                with self.assertRaises(ValueError):
                    allocate(total, weights)
