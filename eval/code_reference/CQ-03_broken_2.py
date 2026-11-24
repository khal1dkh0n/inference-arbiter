"""Seeded bug 2 for CQ-03 (eval/code_questions.yaml): the amount goes through binary floating point.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

import re
from decimal import ROUND_HALF_EVEN, Decimal

_AMOUNT = re.compile(r"-?[0-9]+(\.[0-9]+)?")


def to_minor_units(amount: str, exponent: int) -> int:
    if not 0 <= exponent <= 4:
        raise ValueError("exponent must be between 0 and 4")
    if _AMOUNT.fullmatch(amount) is None:
        raise ValueError("amount is not a plain decimal number")
    return round(float(amount) * 10 ** exponent)
