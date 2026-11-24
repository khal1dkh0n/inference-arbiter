"""Reference tests for CQ-02 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import parse_tlv


class ParseTlvTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(parse_tlv(b""), [])

    def test_one_and_two_byte_tags(self):
        data = bytes.fromhex("5A0841111111111111119F0206000000001234")
        self.assertEqual(
            parse_tlv(data),
            [("5A", bytes.fromhex("4111111111111111")), ("9F02", bytes.fromhex("000000001234"))],
        )

    def test_three_byte_tag(self):
        self.assertEqual(parse_tlv(bytes.fromhex("DF810102ABCD")), [("DF8101", bytes.fromhex("ABCD"))])

    def test_zero_length_value(self):
        self.assertEqual(parse_tlv(bytes.fromhex("9F36005A0112")), [("9F36", b""), ("5A", b"\x12")])

    def test_long_form_one_byte(self):
        value = bytes(range(130))
        self.assertEqual(parse_tlv(bytes.fromhex("7081") + bytes([130]) + value), [("70", value)])

    def test_long_form_two_bytes_is_big_endian(self):
        value = b"\x5a" * 300
        self.assertEqual(parse_tlv(bytes.fromhex("9F4682012C") + value), [("9F46", value)])

    def test_constructed_object_is_not_descended(self):
        inner = bytes.fromhex("5A0212349F020101")
        self.assertEqual(parse_tlv(bytes([0x70, len(inner)]) + inner), [("70", inner)])

    def test_padding_between_and_after_objects(self):
        self.assertEqual(
            parse_tlv(bytes.fromhex("00005A011100009F02012200")),
            [("5A", b"\x11"), ("9F02", b"\x22")],
        )

    def test_truncated_data_raises(self):
        for bad in ("5A", "9F", "DF81", "5A81", "5A8201", "5A0211", "9F020600"):
            with self.subTest(data=bad):
                with self.assertRaises(ValueError):
                    parse_tlv(bytes.fromhex(bad))

    def test_unsupported_length_forms_raise(self):
        for bad in ("5A80", "5A8300000111", "5AFF"):
            with self.subTest(data=bad):
                with self.assertRaises(ValueError):
                    parse_tlv(bytes.fromhex(bad))
