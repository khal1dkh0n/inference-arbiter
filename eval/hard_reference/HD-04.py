"""HD-04: last filing date of a dispute chain with calendar-day and business-day limits."""

from datetime import date, timedelta

QUESTION_DATA = ["2027-10-21", "2028-03-20", "2028-04-14", "2028-04-17", "2028-05-01", "120 calendar days", "30 calendar days", "20 business days"]

HOLIDAYS = {date.fromisoformat(text) for text in QUESTION_DATA[1:5]}


def is_business_day(day: date) -> bool:
    return day.weekday() < 5 and day not in HOLIDAYS


def main() -> None:
    processing_date = date.fromisoformat(QUESTION_DATA[0])
    assert processing_date.strftime("%A") == "Thursday"  # the anchor weekday given in the question
    chargeback = processing_date + timedelta(days=120)
    second_presentment = chargeback + timedelta(days=30)
    while not is_business_day(second_presentment):
        second_presentment += timedelta(days=1)
    pre_arbitration = second_presentment
    counted = 0
    while counted < 20:
        pre_arbitration += timedelta(days=1)
        counted += is_business_day(pre_arbitration)
    arbitration = pre_arbitration + timedelta(days=30)
    while not is_business_day(arbitration):
        arbitration -= timedelta(days=1)
    print(arbitration.isoformat())


if __name__ == "__main__":
    main()
