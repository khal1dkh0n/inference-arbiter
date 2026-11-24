"""Seeded bug 1 for CQ-01 (eval/code_questions.yaml): the wrong digits are doubled.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

def luhn_check_digit(payload: str) -> int:
    if not payload or not (payload.isascii() and payload.isdigit()):
        raise ValueError("payload must be a non-empty string of digits")
    total = 0
    for position, character in enumerate(reversed(payload)):
        digit = int(character)
        if position % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return (10 - total % 10) % 10
