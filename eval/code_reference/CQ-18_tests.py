"""Reference tests for CQ-18 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import wrap_text


class WrapTextTests(unittest.TestCase):
    def test_greedy_fill(self):
        self.assertEqual(wrap_text("the quick brown fox", 10), ["the quick", "brown fox"])
        self.assertEqual(wrap_text("a bb ccc dddd", 6), ["a bb", "ccc", "dddd"])

    def test_exact_fit_stays_on_one_line(self):
        self.assertEqual(wrap_text("ab cd", 5), ["ab cd"])
        self.assertEqual(wrap_text("ab cd", 4), ["ab", "cd"])
        self.assertEqual(wrap_text("abcd", 4), ["abcd"])

    def test_long_word_is_cut(self):
        self.assertEqual(wrap_text("abcdefghij", 4), ["abcd", "efgh", "ij"])

    def test_long_word_ends_the_current_line(self):
        self.assertEqual(wrap_text("x abcdefghij y", 4), ["x", "abcd", "efgh", "ij y"])

    def test_last_piece_of_a_long_word_can_be_joined(self):
        self.assertEqual(wrap_text("abcdef gh", 4), ["abcd", "ef", "gh"])
        self.assertEqual(wrap_text("abcde f", 4), ["abcd", "e f"])

    def test_long_word_that_is_a_multiple_of_the_width(self):
        self.assertEqual(wrap_text("abcdefgh z", 4), ["abcd", "efgh", "z"])

    def test_whitespace_is_collapsed(self):
        self.assertEqual(wrap_text("  a \n b\t c ", 3), ["a b", "c"])

    def test_width_one(self):
        self.assertEqual(wrap_text("ab c", 1), ["a", "b", "c"])

    def test_no_words(self):
        self.assertEqual(wrap_text("", 5), [])
        self.assertEqual(wrap_text("  \n\t ", 5), [])

    def test_width_below_one_raises(self):
        for bad in (0, -3):
            with self.subTest(width=bad):
                with self.assertRaises(ValueError):
                    wrap_text("a", bad)
