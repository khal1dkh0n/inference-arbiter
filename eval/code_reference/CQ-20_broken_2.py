"""Seeded bug 2 for CQ-20 (eval/code_questions.yaml): a single missing integer is written as a run.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

def missing_ranges(numbers: list[int], lower: int, upper: int) -> list[str]:
    if lower > upper:
        raise ValueError("lower is greater than upper")
    ranges: list[str] = []
    expected = lower
    for number in list(numbers) + [upper + 1]:
        if number > expected:
            first, last = expected, number - 1
            ranges.append(f"{first}->{last}")
        expected = number + 1
    return ranges
