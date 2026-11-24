"""Reference tests for CQ-08 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import bitmap_fields


class BitmapFieldsTests(unittest.TestCase):
    def test_typical_primary_bitmap(self):
        self.assertEqual(
            bitmap_fields("7234054128C28805"),
            [2, 3, 4, 7, 11, 12, 14, 22, 24, 26, 32, 35, 37, 41, 42, 47, 49, 53, 62, 64],
        )

    def test_single_bits(self):
        self.assertEqual(bitmap_fields("4000000000000000"), [2])
        self.assertEqual(bitmap_fields("0000000000000001"), [64])
        self.assertEqual(bitmap_fields("0100000000000000"), [8])
        self.assertEqual(bitmap_fields("0080000000000000"), [9])

    def test_no_fields(self):
        self.assertEqual(bitmap_fields("0000000000000000"), [])

    def test_secondary_bitmap(self):
        self.assertEqual(bitmap_fields("80000000000000008000000000000001"), [65, 128])
        self.assertEqual(bitmap_fields("C0000000000000000000000000000000"), [2])
        self.assertEqual(bitmap_fields("F2000000000000000000000004000000"), [2, 3, 4, 7, 102])

    def test_lowercase_hex(self):
        self.assertEqual(bitmap_fields("7234054128c28805"), bitmap_fields("7234054128C28805"))

    def test_wrong_lengths_raise(self):
        for bad in ("", "723405412", "7234054128C2880", "7234054128C288050", "72340541 28C28805"):
            with self.subTest(bitmap=bad):
                with self.assertRaises(ValueError):
                    bitmap_fields(bad)

    def test_bit_one_and_length_must_agree(self):
        with self.assertRaises(ValueError):
            bitmap_fields("8000000000000000")
        with self.assertRaises(ValueError):
            bitmap_fields("40000000000000000000000000000000")

    def test_non_hex_characters_raise(self):
        with self.assertRaises(ValueError):
            bitmap_fields("723405412GC28805")
