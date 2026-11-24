"""Reference tests for CQ-16 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import parse_duration


class ParseDurationTests(unittest.TestCase):
    def test_single_units(self):
        self.assertEqual(parse_duration("90s"), 90)
        self.assertEqual(parse_duration("5m"), 300)
        self.assertEqual(parse_duration("3h"), 10800)
        self.assertEqual(parse_duration("2d"), 172800)

    def test_combined_units(self):
        self.assertEqual(parse_duration("1h30m"), 5400)
        self.assertEqual(parse_duration("1d2h3m4s"), 93784)
        self.assertEqual(parse_duration("1d4s"), 86404)
        self.assertEqual(parse_duration("2h5s"), 7205)

    def test_zero_and_large_numbers(self):
        self.assertEqual(parse_duration("0s"), 0)
        self.assertEqual(parse_duration("0d0h0m0s"), 0)
        self.assertEqual(parse_duration("100m"), 6000)
        self.assertEqual(parse_duration("007s"), 7)

    def test_wrong_order_or_repeats_raise(self):
        for bad in ("1h1h", "30m1h", "1s1m", "1h2d", "1m1m1s"):
            with self.subTest(text=bad):
                with self.assertRaises(ValueError):
                    parse_duration(bad)

    def test_other_text_raises(self):
        for bad in ("", "10", "h", "1.5h", "1H", " 1h", "1h ", "-1h", "1h 30m", "1w", "h1", "1hm", "+1h"):
            with self.subTest(text=bad):
                with self.assertRaises(ValueError):
                    parse_duration(bad)
