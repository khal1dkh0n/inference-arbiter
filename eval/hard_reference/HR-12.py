"""HR-12: largest total value of queued payments that can settle without any overdraft."""

from functools import lru_cache

QUESTION_DATA = [
    "A 50", "B 60", "C 10", "D 80", "E 40",
    "A pays B 30", "C pays D 50", "D pays A 110", "B pays A 50", "D pays C 40", "D pays B 30",
    "B pays D 40", "B pays A 20", "B pays E 40", "B pays C 70", "B pays E 40", "B pays D 60",
]


def main() -> None:
    opening = {line.split()[0]: int(line.split()[1]) for line in QUESTION_DATA[:5]}
    payments = [(line.split()[0], line.split()[2], int(line.split()[3])) for line in QUESTION_DATA[5:]]

    @lru_cache(maxsize=None)
    def best(settled: int) -> int:
        balance = dict(opening)
        value = 0
        for index, (payer, payee, amount) in enumerate(payments):
            if settled >> index & 1:
                balance[payer] -= amount
                balance[payee] += amount
                value += amount
        for index, (payer, _payee, amount) in enumerate(payments):
            if not settled >> index & 1 and balance[payer] >= amount:
                value = max(value, best(settled | 1 << index))
        return value

    print(best(0))


if __name__ == "__main__":
    main()
