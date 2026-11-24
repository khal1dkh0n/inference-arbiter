"""Reference tests for CQ-15 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import decode_varint


class DecodeVarintTests(unittest.TestCase):
    def test_single_byte_values(self):
        self.assertEqual(decode_varint(b"\x00"), (0, 1))
        self.assertEqual(decode_varint(b"\x01"), (1, 1))
        self.assertEqual(decode_varint(b"\x7f"), (127, 1))

    def test_two_byte_values(self):
        self.assertEqual(decode_varint(b"\x80\x01"), (128, 2))
        self.assertEqual(decode_varint(b"\xac\x02"), (300, 2))
        self.assertEqual(decode_varint(b"\x96\x01"), (150, 2))

    def test_five_bytes(self):
        self.assertEqual(decode_varint(b"\xff\xff\xff\xff\x0f"), (4294967295, 5))

    def test_ten_bytes_is_the_longest(self):
        self.assertEqual(decode_varint(b"\xff" * 9 + b"\x01"), (2**64 - 1, 10))

    def test_offset_and_next_offset(self):
        self.assertEqual(decode_varint(b"\x08\x96\x01", 1), (150, 3))
        self.assertEqual(decode_varint(b"\x08\x96\x01\x7f", 3), (127, 4))

    def test_trailing_bytes_are_ignored(self):
        self.assertEqual(decode_varint(b"\x01\xff\xff"), (1, 1))

    def test_truncated_data_raises(self):
        for data, offset in ((b"", 0), (b"\x80", 0), (b"\xff\xff", 0), (b"\x01", 1), (b"\x01", 5)):
            with self.subTest(data=data, offset=offset):
                with self.assertRaises(ValueError):
                    decode_varint(data, offset)

    def test_more_than_ten_bytes_raises(self):
        with self.assertRaises(ValueError):
            decode_varint(b"\x80" * 10 + b"\x01")
