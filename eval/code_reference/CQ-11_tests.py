"""Reference tests for CQ-11 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import unittest

from solution import eval_rpn


class EvalRpnTests(unittest.TestCase):
    def test_simple_expressions(self):
        self.assertEqual(eval_rpn(["2", "1", "+", "3", "*"]), 9)
        self.assertEqual(eval_rpn(["4", "13", "5", "/", "+"]), 6)
        self.assertEqual(eval_rpn(["5"]), 5)
        self.assertEqual(eval_rpn(["-5"]), -5)

    def test_operand_order(self):
        self.assertEqual(eval_rpn(["3", "4", "-"]), -1)
        self.assertEqual(eval_rpn(["20", "5", "/"]), 4)

    def test_division_truncates_toward_zero(self):
        self.assertEqual(eval_rpn(["-7", "2", "/"]), -3)
        self.assertEqual(eval_rpn(["7", "-2", "/"]), -3)
        self.assertEqual(eval_rpn(["-7", "-2", "/"]), 3)
        self.assertEqual(eval_rpn(["7", "2", "/"]), 3)
        self.assertEqual(eval_rpn(["-1", "2", "/"]), 0)

    def test_longer_expression(self):
        tokens = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
        self.assertEqual(eval_rpn(tokens), 22)

    def test_large_integers_are_exact(self):
        self.assertEqual(eval_rpn(["100000000000000000000", "3", "/"]), 33333333333333333333)
        self.assertEqual(eval_rpn(["-100000000000000000001", "2", "/"]), -50000000000000000000)

    def test_division_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            eval_rpn(["1", "0", "/"])

    def test_malformed_expressions_raise(self):
        for bad in ([], ["+"], ["1", "+"], ["1", "2"], ["1", "2", "x"], ["1.5"], ["1", "2", "+", "+"], [""], ["+1"]):
            with self.subTest(tokens=bad):
                with self.assertRaises(ValueError):
                    eval_rpn(bad)
