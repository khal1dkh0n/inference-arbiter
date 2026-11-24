"""Reference tests for CQ-07 (eval/code_questions.yaml).

Used for grading only. They are never part of a prompt.
"""

import datetime
import unittest

from solution import add_business_days

D = datetime.date


class AddBusinessDaysTests(unittest.TestCase):
    def test_calendar_assumptions(self):
        self.assertEqual(D(2026, 1, 2).weekday(), 4)  # a Friday
        self.assertEqual(D(2025, 12, 31).weekday(), 2)  # a Wednesday

    def test_friday_plus_one_is_monday(self):
        self.assertEqual(add_business_days(D(2026, 1, 2), 1, set()), D(2026, 1, 5))

    def test_a_full_week(self):
        self.assertEqual(add_business_days(D(2026, 1, 5), 5, set()), D(2026, 1, 12))

    def test_start_on_a_saturday(self):
        self.assertEqual(add_business_days(D(2026, 1, 3), 1, set()), D(2026, 1, 5))
        self.assertEqual(add_business_days(D(2026, 1, 3), 2, set()), D(2026, 1, 6))

    def test_holiday_is_skipped(self):
        self.assertEqual(add_business_days(D(2026, 1, 2), 1, {D(2026, 1, 5)}), D(2026, 1, 6))

    def test_across_the_year_end_with_a_holiday(self):
        holidays = {D(2026, 1, 1)}
        self.assertEqual(add_business_days(D(2025, 12, 31), 1, holidays), D(2026, 1, 2))
        self.assertEqual(add_business_days(D(2025, 12, 31), 3, holidays), D(2026, 1, 6))

    def test_holiday_on_a_weekend_changes_nothing(self):
        self.assertEqual(add_business_days(D(2026, 1, 2), 1, {D(2026, 1, 3)}), D(2026, 1, 5))

    def test_start_is_not_counted_even_when_it_is_a_holiday(self):
        self.assertEqual(add_business_days(D(2026, 1, 5), 1, {D(2026, 1, 5)}), D(2026, 1, 6))

    def test_zero_days_returns_start_unchanged(self):
        self.assertEqual(add_business_days(D(2026, 1, 3), 0, set()), D(2026, 1, 3))
        self.assertEqual(add_business_days(D(2026, 1, 5), 0, {D(2026, 1, 5)}), D(2026, 1, 5))

    def test_negative_days_raise(self):
        with self.assertRaises(ValueError):
            add_business_days(D(2026, 1, 5), -1, set())
