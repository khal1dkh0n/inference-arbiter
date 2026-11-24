"""Reference tests for CQ-04 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import crc16


class Crc16Tests(unittest.TestCase):
    def test_check_value(self):
        self.assertEqual(crc16(b"123456789"), 0x29B1)

    def test_empty_data(self):
        self.assertEqual(crc16(b""), 0xFFFF)

    def test_single_bytes(self):
        self.assertEqual(crc16(b"A"), 0xB915)
        self.assertEqual(crc16(b"\x00"), 0xE1F0)

    def test_longer_input(self):
        self.assertEqual(crc16(bytes(range(256))), 0x3FBD)

    def test_result_fits_sixteen_bits(self):
        value = crc16(b"\xff" * 64)
        self.assertIsInstance(value, int)
        self.assertTrue(0 <= value <= 0xFFFF)
