"""Reference tests for CQ-10 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import top_k_words


class TopKWordsTests(unittest.TestCase):
    def test_most_frequent_first(self):
        self.assertEqual(
            top_k_words("the cat and the hat and the bat", 2), [("the", 3), ("and", 2)]
        )

    def test_ties_are_alphabetical(self):
        self.assertEqual(
            top_k_words("pear apple pear apple fig", 3), [("apple", 2), ("pear", 2), ("fig", 1)]
        )
        self.assertEqual(top_k_words("zebra yak ant", 2), [("ant", 1), ("yak", 1)])

    def test_case_is_ignored_and_words_are_lowercase(self):
        self.assertEqual(top_k_words("The the THE Cat", 2), [("the", 3), ("cat", 1)])

    def test_non_letters_separate_words(self):
        self.assertEqual(
            top_k_words("it's 2nd-rate, it is", 10),
            [("it", 2), ("is", 1), ("nd", 1), ("rate", 1), ("s", 1)],
        )

    def test_k_larger_than_the_vocabulary(self):
        self.assertEqual(top_k_words("b a", 5), [("a", 1), ("b", 1)])

    def test_k_zero_or_negative(self):
        self.assertEqual(top_k_words("a b c", 0), [])
        self.assertEqual(top_k_words("a b c", -2), [])

    def test_no_words(self):
        self.assertEqual(top_k_words("", 3), [])
        self.assertEqual(top_k_words("123 ... !!!", 3), [])

    def test_items_are_tuples(self):
        self.assertIsInstance(top_k_words("a", 1)[0], tuple)
