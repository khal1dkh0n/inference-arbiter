"""Seeded bug 1 for CQ-11 (eval/code_questions.yaml): division floors, it does not truncate.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

import re

_INTEGER = re.compile(r"-?[0-9]+")


def _divide(dividend: int, divisor: int) -> int:
    if divisor == 0:
        raise ZeroDivisionError("division by zero")
    return dividend // divisor


def eval_rpn(tokens: list[str]) -> int:
    stack: list[int] = []
    for token in tokens:
        if token in ("+", "-", "*", "/"):
            if len(stack) < 2:
                raise ValueError("an operator needs two values")
            right = stack.pop()
            left = stack.pop()
            if token == "+":
                stack.append(left + right)
            elif token == "-":
                stack.append(left - right)
            elif token == "*":
                stack.append(left * right)
            else:
                stack.append(_divide(left, right))
        elif _INTEGER.fullmatch(token):
            stack.append(int(token))
        else:
            raise ValueError("unknown token")
    if len(stack) != 1:
        raise ValueError("the expression does not reduce to one value")
    return stack[0]
