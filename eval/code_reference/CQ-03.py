"""Reference solution for CQ-03 (eval/code_questions.yaml).

It shows that the reference tests pass a right solution. It is never part of a prompt.
"""

import re
from decimal import ROUND_HALF_EVEN, Decimal

_AMOUNT = re.compile(r"-?[0-9]+(\.[0-9]+)?")


def to_minor_units(amount: str, exponent: int) -> int:
    if not 0 <= exponent <= 4:
        raise ValueError("exponent must be between 0 and 4")
    if _AMOUNT.fullmatch(amount) is None:
        raise ValueError("amount is not a plain decimal number")
    scaled = Decimal(amount).scaleb(exponent)
    return int(scaled.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
