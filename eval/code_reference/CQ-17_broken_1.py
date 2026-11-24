"""Seeded bug 1 for CQ-17 (eval/code_questions.yaml): numeric identifiers are compared as text.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

def _parse(version: str) -> tuple[list[int], list[str]]:
    without_build = version.split("+", 1)[0]
    core, separator, prerelease = without_build.partition("-")
    numbers = [int(part) for part in core.split(".")]
    identifiers = prerelease.split(".") if separator else []
    return numbers, identifiers


def _compare(left, right) -> int:
    if left == right:
        return 0
    return -1 if left < right else 1


def _compare_identifiers(left: str, right: str) -> int:
    if left.isdigit() and right.isdigit():
        return _compare(left, right)
    if left.isdigit():
        return -1
    if right.isdigit():
        return 1
    return _compare(left, right)


def compare_versions(a: str, b: str) -> int:
    numbers_a, identifiers_a = _parse(a)
    numbers_b, identifiers_b = _parse(b)
    if numbers_a != numbers_b:
        return _compare(numbers_a, numbers_b)
    if not identifiers_a or not identifiers_b:
        return _compare(bool(identifiers_b), bool(identifiers_a))
    for left, right in zip(identifiers_a, identifiers_b):
        result = _compare_identifiers(left, right)
        if result != 0:
            return result
    return _compare(len(identifiers_a), len(identifiers_b))
