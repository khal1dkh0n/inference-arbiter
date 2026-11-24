"""Seeded bug 1 for CQ-07 (eval/code_questions.yaml): Saturday counts as a business day.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

import datetime


def add_business_days(start: datetime.date, days: int, holidays: set[datetime.date]) -> datetime.date:
    if days < 0:
        raise ValueError("days must not be negative")
    current = start
    remaining = days
    while remaining > 0:
        current += datetime.timedelta(days=1)
        if current.weekday() < 6 and current not in holidays:
            remaining -= 1
    return current
