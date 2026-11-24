"""HR-13: number of serial orders that are conflict-equivalent to a six-transaction schedule."""

from itertools import permutations

QUESTION_DATA = [
    "T3 W(E)", "T3 W(D)", "T5 W(C)", "T5 R(D)", "T6 R(A)", "T4 R(B)", "T3 R(A)", "T6 R(B)", "T4 R(D)",
    "T1 R(D)", "T6 R(E)", "T1 W(E)", "T2 R(D)", "T4 W(B)", "T1 R(E)", "T2 R(E)", "T5 R(E)",
]


def main() -> None:
    schedule = [(line.split()[0], line.split()[1][0], line.split()[1][2]) for line in QUESTION_DATA]
    before = set()
    for position, (txn, kind, item) in enumerate(schedule):
        for later_txn, later_kind, later_item in schedule[position + 1:]:
            if txn != later_txn and item == later_item and "W" in (kind, later_kind):
                before.add((txn, later_txn))
    transactions = sorted({txn for txn, _kind, _item in schedule})
    print(sum(
        1 for order in permutations(transactions)
        if all(order.index(first) < order.index(second) for first, second in before)
    ))


if __name__ == "__main__":
    main()
