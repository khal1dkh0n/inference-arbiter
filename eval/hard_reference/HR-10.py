"""HR-10: cheapest routing of nine transactions across three acquirers, by exhaustive search."""

from itertools import product

QUESTION_DATA = [
    "1  800.00  debit", "2  350.00  credit", "3  800.00  credit", "4  1,250.00  credit", "5  750.00  debit",
    "6  250.00  debit", "7  750.00  credit", "8  100.00  debit", "9  1,450.00  debit",
    "1.80%", "0.60%", "0.10", "2,500.00", "1.50%", "0.90%", "0.25", "3,000.00", "0.30", "1.20%", "30.00",
]


def cost(assignment: tuple[str, ...], transactions: list[tuple[int, str]]) -> int | None:
    """Total fee in cents, or None if the routing breaks acquirer A's cap. Rates are in basis points."""
    total = 0
    for acquirer in "ABC":
        mine = [txn for txn, chosen in zip(transactions, assignment) if chosen == acquirer]
        if not mine:
            continue
        volume = sum(amount for amount, _kind in mine)
        if acquirer == "A":
            if volume > 250_000:
                return None
            total += sum(amount * (180 if kind == "credit" else 60) // 10_000 + 10 for amount, kind in mine)
        elif acquirer == "B":
            discount = 30 if volume >= 300_000 else 0
            total += sum(
                amount * ((150 if kind == "credit" else 90) - discount) // 10_000 + 25 for amount, kind in mine
            )
        else:
            total += max(sum(amount * 120 // 10_000 for amount, _kind in mine), 3_000)
    return total


def main() -> None:
    transactions = []
    for line in QUESTION_DATA[:9]:
        _number, amount, kind = line.split()
        cents = int(amount.replace(",", "").replace(".", ""))
        assert cents * 30 % 10_000 == 0  # every rate is a multiple of 0.30%, so each fee is whole cents
        transactions.append((cents, kind))
    costs = [cost(assignment, transactions) for assignment in product("ABC", repeat=len(transactions))]
    best = min(value for value in costs if value is not None)
    print(f"{best // 100}.{best % 100:02d}")


if __name__ == "__main__":
    main()
