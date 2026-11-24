"""Reference tests for CQ-01 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import luhn_check_digit


class LuhnCheckDigitTests(unittest.TestCase):
    def test_known_numbers(self):
        self.assertEqual(luhn_check_digit("7992739871"), 3)
        self.assertEqual(luhn_check_digit("411111111111111"), 1)
        self.assertEqual(luhn_check_digit("37828224631000"), 5)

    def test_single_digits(self):
        self.assertEqual(luhn_check_digit("1"), 8)
        self.assertEqual(luhn_check_digit("5"), 9)
        self.assertEqual(luhn_check_digit("9"), 1)

    def test_check_digit_zero(self):
        self.assertEqual(luhn_check_digit("0"), 0)
        self.assertEqual(luhn_check_digit("000000"), 0)

    def test_two_digits(self):
        self.assertEqual(luhn_check_digit("18"), 2)

    def test_returns_an_int(self):
        self.assertIsInstance(luhn_check_digit("7992739871"), int)

    def test_rejects_anything_but_digits(self):
        for bad in ("", "12 34", "12a", "-12", "+1", "1.0"):
            with self.subTest(payload=bad):
                with self.assertRaises(ValueError):
                    luhn_check_digit(bad)
