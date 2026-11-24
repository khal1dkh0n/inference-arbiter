"""Reference tests for CQ-14 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import mask_pan


class MaskPanTests(unittest.TestCase):
    def test_plain_sixteen_digits(self):
        self.assertEqual(mask_pan("4111111111111111"), "411111******1111")
        self.assertEqual(mask_pan("5500005555555559"), "550000******5559")

    def test_spaces_stay_in_place(self):
        self.assertEqual(mask_pan("4111 1111 1111 1111"), "4111 11** **** 1111")

    def test_hyphens_stay_in_place(self):
        self.assertEqual(mask_pan("4111-1111-1111-1111"), "4111-11**-****-1111")

    def test_mixed_and_edge_separators(self):
        self.assertEqual(mask_pan(" 4111 1111-1111 1111 "), " 4111 11**-**** 1111 ")

    def test_shortest_and_longest(self):
        self.assertEqual(mask_pan("4222222222222"), "422222***2222")
        self.assertEqual(mask_pan("6011000990139424123"), "601100*********4123")

    def test_fifteen_digits(self):
        self.assertEqual(mask_pan("3782 822463 10005"), "3782 82**** *0005")

    def test_wrong_digit_counts_raise(self):
        for bad in ("", "411111111111", "41111111111111111111", "4111 1111 1111"):
            with self.subTest(pan=bad):
                with self.assertRaises(ValueError):
                    mask_pan(bad)

    def test_other_characters_raise(self):
        for bad in ("4111.1111.1111.1111", "4111111111111111a", "4111_1111_1111_1111"):
            with self.subTest(pan=bad):
                with self.assertRaises(ValueError):
                    mask_pan(bad)
