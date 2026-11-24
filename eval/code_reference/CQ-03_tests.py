"""Reference tests for CQ-03 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import to_minor_units


class ToMinorUnitsTests(unittest.TestCase):
    def test_no_rounding_needed(self):
        self.assertEqual(to_minor_units("12.34", 2), 1234)
        self.assertEqual(to_minor_units("12", 2), 1200)
        self.assertEqual(to_minor_units("12.5", 2), 1250)
        self.assertEqual(to_minor_units("7", 0), 7)
        self.assertEqual(to_minor_units("1.234", 3), 1234)

    def test_half_goes_to_the_even_neighbour(self):
        self.assertEqual(to_minor_units("0.285", 2), 28)
        self.assertEqual(to_minor_units("0.295", 2), 30)
        self.assertEqual(to_minor_units("1.005", 2), 100)
        self.assertEqual(to_minor_units("2.675", 2), 268)
        self.assertEqual(to_minor_units("1234.5", 0), 1234)
        self.assertEqual(to_minor_units("1235.5", 0), 1236)
        self.assertEqual(to_minor_units("1.2345", 3), 1234)
        self.assertEqual(to_minor_units("0.575", 2), 58)

    def test_large_amounts_are_exact(self):
        self.assertEqual(to_minor_units("90071992547409.93", 2), 9007199254740993)
        self.assertEqual(to_minor_units("123456789012345678901234.565", 2), 12345678901234567890123456)

    def test_not_a_half_rounds_to_nearest(self):
        self.assertEqual(to_minor_units("0.0051", 2), 1)
        self.assertEqual(to_minor_units("0.2851", 2), 29)
        self.assertEqual(to_minor_units("0.2849", 2), 28)

    def test_negative_amounts(self):
        self.assertEqual(to_minor_units("-0.125", 2), -12)
        self.assertEqual(to_minor_units("-0.135", 2), -14)
        self.assertEqual(to_minor_units("-12.34", 2), -1234)

    def test_negative_zero_is_zero(self):
        self.assertEqual(to_minor_units("-0.00", 2), 0)
        self.assertEqual(to_minor_units("-0.001", 2), 0)

    def test_returns_an_int(self):
        self.assertIsInstance(to_minor_units("12.34", 2), int)

    def test_rejects_other_formats(self):
        for bad in ("", "1.", ".5", "1,5", " 1", "1 ", "+1", "1e2", "--1", "1.2.3", "abc"):
            with self.subTest(amount=bad):
                with self.assertRaises(ValueError):
                    to_minor_units(bad, 2)

    def test_rejects_exponents_out_of_range(self):
        for bad in (-1, 5):
            with self.subTest(exponent=bad):
                with self.assertRaises(ValueError):
                    to_minor_units("1.00", bad)
