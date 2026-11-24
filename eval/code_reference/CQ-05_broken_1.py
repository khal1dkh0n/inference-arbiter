"""Seeded bug 1 for CQ-05 (eval/code_questions.yaml): adjacent intervals are not merged.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    for start, end in intervals:
        if start > end:
            raise ValueError("interval start is greater than its end")
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
