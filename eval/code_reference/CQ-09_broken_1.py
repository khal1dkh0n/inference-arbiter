"""Seeded bug 1 for CQ-09 (eval/code_questions.yaml): non-canonical numerals are accepted.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

import re

_CANONICAL = re.compile(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})")
_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(text: str) -> int:
    if not text or any(letter not in _VALUES for letter in text):
        raise ValueError("not a canonical Roman numeral")
    total = 0
    for index, letter in enumerate(text):
        value = _VALUES[letter]
        if index + 1 < len(text) and value < _VALUES[text[index + 1]]:
            total -= value
        else:
            total += value
    return total
