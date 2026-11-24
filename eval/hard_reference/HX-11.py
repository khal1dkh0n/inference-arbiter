"""HX-11: a Luhn check-digit function with one bug; buggy outputs then correct outputs."""

QUESTION_DATA = ["79927398719", "4929939187355", "371449635398439", "d = d * 2 % 9"]
PAYLOADS = QUESTION_DATA[:3]


def check_digit(payload):
    total = 0
    for i, ch in enumerate(reversed(payload)):
        d = int(ch)
        if i % 2 == 0:
            d = d * 2 % 9
        total += d
    return (10 - total % 10) % 10


def correct_check_digit(payload):
    total = 0
    for i, ch in enumerate(reversed(payload)):
        d = int(ch)
        if i % 2 == 0:
            d = d * 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def main() -> None:
    buggy = "".join(str(check_digit(payload)) for payload in PAYLOADS)
    correct = "".join(str(correct_check_digit(payload)) for payload in PAYLOADS)
    assert buggy != correct
    print(buggy + correct)


if __name__ == "__main__":
    main()
