"""Reference tests for CQ-06 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import is_valid_iban


class IsValidIbanTests(unittest.TestCase):
    def test_valid_examples(self):
        self.assertIs(is_valid_iban("GB82WEST12345698765432"), True)
        self.assertIs(is_valid_iban("DE89370400440532013000"), True)
        self.assertIs(is_valid_iban("FR1420041010050500013M02606"), True)

    def test_shortest_allowed_length(self):
        self.assertIs(is_valid_iban("NO9386011117947"), True)

    def test_spaces_and_lowercase_are_accepted(self):
        self.assertIs(is_valid_iban("GB82 WEST 1234 5698 7654 32"), True)
        self.assertIs(is_valid_iban("gb82west12345698765432"), True)

    def test_one_changed_character_fails(self):
        self.assertIs(is_valid_iban("GB82WEST12345698765433"), False)
        self.assertIs(is_valid_iban("GB83WEST12345698765432"), False)
        self.assertIs(is_valid_iban("DE89370400440532013001"), False)

    def test_swapped_characters_fail(self):
        self.assertIs(is_valid_iban("GB82WEST12345698765423"), False)

    def test_wrong_shape_fails(self):
        self.assertIs(is_valid_iban(""), False)
        self.assertIs(is_valid_iban("GB82"), False)
        self.assertIs(is_valid_iban("GB82-WEST-1234-5698-7654-32"), False)
        self.assertIs(is_valid_iban("1282WEST12345698765432"), False)
        self.assertIs(is_valid_iban("GBXXWEST12345698765432"), False)
        self.assertIs(is_valid_iban("GB82WEST12345698765432" + "0" * 13), False)
