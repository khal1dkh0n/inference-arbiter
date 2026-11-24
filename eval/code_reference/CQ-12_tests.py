"""Reference tests for CQ-12 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import normalize_path


class NormalizePathTests(unittest.TestCase):
    def check(self, path, expected):
        self.assertEqual(normalize_path(path), expected, msg=repr(path))

    def test_absolute_paths(self):
        self.check("/a/b/../c", "/a/c")
        self.check("/a/./b/", "/a/b")
        self.check("//a//b", "/a/b")
        self.check("/", "/")
        self.check("///", "/")

    def test_parent_of_the_root_is_the_root(self):
        self.check("/../a", "/a")
        self.check("/a/b/../../..", "/")
        self.check("/..", "/")

    def test_relative_paths(self):
        self.check("a/b/../c", "a/c")
        self.check("./a/./b/", "a/b")
        self.check("a//b", "a/b")

    def test_relative_paths_keep_leading_parents(self):
        self.check("a/../../b", "../b")
        self.check("../../a", "../../a")
        self.check("a/b/../../../c", "../c")
        self.check("..", "..")

    def test_nothing_left(self):
        self.check("", ".")
        self.check(".", ".")
        self.check("a/..", ".")
        self.check("./.", ".")

    def test_names_that_only_look_special(self):
        self.check("/a/.../b", "/a/.../b")
        self.check("a/..b/c", "a/..b/c")
