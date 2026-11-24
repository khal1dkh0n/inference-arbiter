"""Reference tests for CQ-20 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import missing_ranges


class MissingRangesTests(unittest.TestCase):
    def test_typical(self):
        self.assertEqual(
            missing_ranges([0, 1, 3, 50, 75], 0, 99), ["2", "4->49", "51->74", "76->99"]
        )

    def test_nothing_given(self):
        self.assertEqual(missing_ranges([], 1, 1), ["1"])
        self.assertEqual(missing_ranges([], 1, 5), ["1->5"])

    def test_nothing_missing(self):
        self.assertEqual(missing_ranges([1, 2, 3], 1, 3), [])
        self.assertEqual(missing_ranges([7], 7, 7), [])

    def test_missing_at_the_ends(self):
        self.assertEqual(missing_ranges([5], 5, 6), ["6"])
        self.assertEqual(missing_ranges([5], 4, 5), ["4"])
        self.assertEqual(missing_ranges([5], 1, 9), ["1->4", "6->9"])

    def test_run_of_two(self):
        self.assertEqual(missing_ranges([1, 4], 1, 4), ["2->3"])

    def test_negative_numbers(self):
        self.assertEqual(missing_ranges([-3, 0], -5, 2), ["-5->-4", "-2->-1", "1->2"])

    def test_lower_above_upper_raises(self):
        with self.assertRaises(ValueError):
            missing_ranges([], 3, 2)

    def test_input_is_not_modified(self):
        numbers = [2, 4]
        missing_ranges(numbers, 1, 5)
        self.assertEqual(numbers, [2, 4])
