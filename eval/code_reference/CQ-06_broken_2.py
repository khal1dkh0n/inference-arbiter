"""Seeded bug 2 for CQ-06 (eval/code_questions.yaml): the remainder is compared with 0.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

import string

_ALLOWED = set(string.ascii_uppercase + string.digits)


def is_valid_iban(iban: str) -> bool:
    compact = iban.replace(" ", "").upper()
    if not 15 <= len(compact) <= 34:
        return False
    if any(character not in _ALLOWED for character in compact):
        return False
    if not compact[:2].isalpha() or not compact[2:4].isdigit():
        return False
    rearranged = compact[4:] + compact[:4]
    number = "".join(str(int(character, 36)) for character in rearranged)
    return int(number) % 97 == 0
