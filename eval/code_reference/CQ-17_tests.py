"""Reference tests for CQ-17 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import compare_versions


class CompareVersionsTests(unittest.TestCase):
    def check(self, lower, higher):
        self.assertEqual(compare_versions(lower, higher), -1, msg=f"{lower} < {higher}")
        self.assertEqual(compare_versions(higher, lower), 1, msg=f"{higher} > {lower}")

    def test_core_numbers(self):
        self.check("1.0.0", "2.0.0")
        self.check("2.0.9", "2.1.0")
        self.check("2.1.0", "2.1.1")

    def test_numbers_are_not_compared_as_text(self):
        self.check("1.9.0", "1.10.0")
        self.check("9.0.0", "10.0.0")
        self.check("1.0.9", "1.0.10")

    def test_pre_release_is_lower_than_the_release(self):
        self.check("1.0.0-alpha", "1.0.0")
        self.check("1.0.0-rc.1", "1.0.0")
        self.check("1.0.0-0", "1.0.0")

    def test_the_documented_precedence_chain(self):
        chain = [
            "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
            "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0",
        ]
        for lower, higher in zip(chain, chain[1:]):
            self.check(lower, higher)

    def test_numeric_identifier_is_lower_than_a_named_one(self):
        self.check("1.0.0-1", "1.0.0-alpha")
        self.check("1.0.0-alpha.9", "1.0.0-alpha.a")

    def test_identifiers_with_hyphens(self):
        self.check("1.0.0-beta-1", "1.0.0-beta-2")
        self.check("1.0.0-9", "1.0.0-x-1")

    def test_build_metadata_is_ignored(self):
        self.assertEqual(compare_versions("1.0.0+build1", "1.0.0+build2"), 0)
        self.assertEqual(compare_versions("1.0.0-alpha+x", "1.0.0-alpha"), 0)
        self.check("1.0.0-alpha+zzz", "1.0.0+aaa")

    def test_equal_versions(self):
        self.assertEqual(compare_versions("1.2.3", "1.2.3"), 0)
        self.assertEqual(compare_versions("1.2.3-rc.1", "1.2.3-rc.1"), 0)
