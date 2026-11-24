"""HX-10: what a float-based minor-unit conversion returns for a list of amounts."""

QUESTION_DATA = ['"19.99", "4.35", "1.15", "0.29", "8.20", "2.50", "1.10", "16.90", "0.57", "1.13"']
AMOUNTS = ["19.99", "4.35", "1.15", "0.29", "8.20", "2.50", "1.10", "16.90", "0.57", "1.13"]


def to_minor_units(amount):
    return int(float(amount) * 100)


def batch_total(amounts):
    return sum(to_minor_units(amount) for amount in amounts)


def main() -> None:
    assert ", ".join(f'"{amount}"' for amount in AMOUNTS) == QUESTION_DATA[0]
    print(batch_total(AMOUNTS))


if __name__ == "__main__":
    main()
