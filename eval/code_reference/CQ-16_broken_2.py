"""Seeded bug 2 for CQ-16 (eval/code_questions.yaml): the empty string is zero seconds.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

import re

_DURATION = re.compile(r"(?:([0-9]+)d)?(?:([0-9]+)h)?(?:([0-9]+)m)?(?:([0-9]+)s)?")
_SECONDS = (86400, 3600, 60, 1)


def parse_duration(text: str) -> int:
    match = _DURATION.fullmatch(text)
    if match is None:
        raise ValueError("not a duration")
    return sum(int(part) * seconds for part, seconds in zip(match.groups(), _SECONDS) if part is not None)
