"""Reference solution for CQ-20 (eval/code_questions.yaml).

It shows that the reference tests pass a right solution. It is never part of a prompt.
"""

def missing_ranges(numbers: list[int], lower: int, upper: int) -> list[str]:
    if lower > upper:
        raise ValueError("lower is greater than upper")
    ranges: list[str] = []
    expected = lower
    for number in list(numbers) + [upper + 1]:
        if number > expected:
            first, last = expected, number - 1
            ranges.append(str(first) if first == last else f"{first}->{last}")
        expected = number + 1
    return ranges
