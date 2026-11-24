"""Reference tests for CQ-05 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import merge_intervals


class MergeIntervalsTests(unittest.TestCase):
    def test_overlapping(self):
        self.assertEqual(
            merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)]), [(1, 6), (8, 10), (15, 18)]
        )

    def test_adjacent_intervals_merge(self):
        self.assertEqual(merge_intervals([(1, 2), (3, 4)]), [(1, 4)])

    def test_a_gap_of_one_integer_stays(self):
        self.assertEqual(merge_intervals([(1, 2), (4, 5)]), [(1, 2), (4, 5)])

    def test_unsorted_input(self):
        self.assertEqual(merge_intervals([(8, 10), (1, 3), (2, 6)]), [(1, 6), (8, 10)])

    def test_contained_interval(self):
        self.assertEqual(merge_intervals([(1, 10), (2, 3)]), [(1, 10)])
        self.assertEqual(merge_intervals([(1, 10), (2, 3), (4, 12)]), [(1, 12)])

    def test_duplicates_and_single_points(self):
        self.assertEqual(merge_intervals([(5, 5), (5, 5), (6, 6)]), [(5, 6)])

    def test_negative_numbers(self):
        self.assertEqual(merge_intervals([(-5, -3), (-2, 0), (2, 2)]), [(-5, 0), (2, 2)])

    def test_empty_and_single(self):
        self.assertEqual(merge_intervals([]), [])
        self.assertEqual(merge_intervals([(4, 7)]), [(4, 7)])

    def test_start_after_end_raises(self):
        with self.assertRaises(ValueError):
            merge_intervals([(1, 2), (5, 4)])

    def test_input_is_not_modified(self):
        intervals = [(8, 10), (1, 3), (2, 6)]
        merge_intervals(intervals)
        self.assertEqual(intervals, [(8, 10), (1, 3), (2, 6)])
