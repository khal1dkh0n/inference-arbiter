"""Seeded bug 2 for CQ-08 (eval/code_questions.yaml): bits are numbered from the least significant end.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

import string


def bitmap_fields(bitmap_hex: str) -> list[int]:
    if len(bitmap_hex) not in (16, 32) or any(character not in string.hexdigits for character in bitmap_hex):
        raise ValueError("a bitmap is 16 or 32 hex characters")
    bits = bin(int(bitmap_hex, 16))[2:].zfill(len(bitmap_hex) * 4)
    has_secondary = bits[0] == "1"
    if has_secondary != (len(bitmap_hex) == 32):
        raise ValueError("bit 1 and the bitmap length disagree")
    return [index + 1 for index, bit in enumerate(reversed(bits)) if bit == "1" and index != 0]
