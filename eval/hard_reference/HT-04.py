"""HT-04: check a settlement batch's trailer against its detail rows."""

QUESTION_DATA = [
    "01 D 412.80", "02 D 96.15", "03 D 1,204.00", "04 C 318.45", "05 D 57.30", "06 D 733.90", "07 D 240.00",
    "08 D 88.75", "09 C 142.60", "10 D 615.25", "11 D 39.99", "12 D 1,050.00", "13 D 264.10", "14 D 507.20",
    "TRAILER count 14 total 5,770.49",
]


def cents(text: str) -> int:
    return int(text.replace(",", "").replace(".", ""))


def main() -> None:
    rows = [line.split() for line in QUESTION_DATA[:-1]]
    net = sum(cents(amount) if kind == "D" else -cents(amount) for _number, kind, amount in rows)
    gross = sum(cents(amount) for _number, _kind, amount in rows)
    trailer = QUESTION_DATA[-1].split()
    trailer_count, trailer_total = int(trailer[2]), cents(trailer[4])
    assert gross == trailer_total  # the trailer was built by adding the credits instead of subtracting them
    verdict = "MATCH" if (len(rows), net) == (trailer_count, trailer_total) else "MISMATCH"
    print(f"count={len(rows)} net={net / 100:.2f} trailer={trailer_total / 100:.2f} {verdict}")


if __name__ == "__main__":
    main()
