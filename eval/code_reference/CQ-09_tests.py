"""Reference tests for CQ-09 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import roman_to_int


class RomanToIntTests(unittest.TestCase):
    def test_single_letters(self):
        for text, value in (("I", 1), ("V", 5), ("X", 10), ("L", 50), ("C", 100), ("D", 500), ("M", 1000)):
            with self.subTest(text=text):
                self.assertEqual(roman_to_int(text), value)

    def test_subtractive_pairs(self):
        for text, value in (("IV", 4), ("IX", 9), ("XL", 40), ("XC", 90), ("CD", 400), ("CM", 900)):
            with self.subTest(text=text):
                self.assertEqual(roman_to_int(text), value)

    def test_longer_numerals(self):
        self.assertEqual(roman_to_int("XIV"), 14)
        self.assertEqual(roman_to_int("XCIX"), 99)
        self.assertEqual(roman_to_int("DCCCLXXXVIII"), 888)
        self.assertEqual(roman_to_int("MCMXC"), 1990)
        self.assertEqual(roman_to_int("MMXXVI"), 2026)
        self.assertEqual(roman_to_int("MMMCMXCIX"), 3999)

    def test_non_canonical_forms_raise(self):
        for bad in ("IIII", "VV", "IC", "XM", "IVI", "MMMM", "IIV", "VX", "LL", "XXXX", "CMCM", "IXI"):
            with self.subTest(text=bad):
                with self.assertRaises(ValueError):
                    roman_to_int(bad)

    def test_other_input_raises(self):
        for bad in ("", "iv", "Xiv", "ABC", "X I", "12", " X"):
            with self.subTest(text=bad):
                with self.assertRaises(ValueError):
                    roman_to_int(bad)
