"""Reference solution for CQ-14 (eval/code_questions.yaml).

It shows that the reference tests pass a right solution. It is never part of a prompt.
"""

def mask_pan(pan: str) -> str:
    if any(character not in "0123456789 -" for character in pan):
        raise ValueError("only digits, spaces and hyphens are allowed")
    digit_count = sum(1 for character in pan if character not in " -")
    if not 13 <= digit_count <= 19:
        raise ValueError("a card number has 13 to 19 digits")
    masked = []
    seen = 0
    for character in pan:
        if character in " -":
            masked.append(character)
            continue
        seen += 1
        if seen <= 6 or seen > digit_count - 4:
            masked.append(character)
        else:
            masked.append("*")
    return "".join(masked)
