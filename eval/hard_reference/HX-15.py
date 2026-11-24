"""HX-15: which of twenty numbers pass the Luhn check; sum of the last four digits of those."""

QUESTION_DATA = [
    "3501117735686", "5596263776164820", "6993937996084", "2129442768946", "340635187430734",
    "3362552682627480274", "39443140731311", "3572672121244910", "39451326605715", "3520818661866891393",
    "6552406993525862", "263249215191228", "2681426349293", "5272197304730819", "6495672206917495304",
    "5158159358259", "6507326541946054", "5760124383339923", "2641124557258307", "2947681864816791",
]


def luhn_valid(number: str) -> bool:
    total = 0
    for index, character in enumerate(reversed(number)):
        digit = int(character)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def main() -> None:
    valid = [number for number in QUESTION_DATA if luhn_valid(number)]
    assert len(valid) == 6
    print(sum(int(number[-4:]) for number in valid))


if __name__ == "__main__":
    main()
