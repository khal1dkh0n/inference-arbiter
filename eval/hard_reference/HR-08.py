"""HR-08: sum of the distinct final balances over every interleaving of three workers."""

from itertools import permutations

QUESTION_DATA = ["starts at 100", "write x + 50", "write y * 2", "write z - 150", "write z + 5", "write z - 20"]

STEPS = {"W1": 2, "W2": 2, "W3": 4}


def run(order: tuple[str, ...]) -> int:
    balance = 100
    local: dict[str, int] = {}
    done = dict.fromkeys(STEPS, 0)
    for worker in order:
        step = done[worker]
        done[worker] += 1
        if step in (0, 2):  # steps 1 and 3 are reads
            local[worker] = balance
        elif worker == "W1":
            balance = local[worker] + 50
        elif worker == "W2":
            balance = local[worker] * 2
        elif step == 1:
            balance = local[worker] - 150 if local[worker] >= 150 else local[worker] + 5
        else:
            balance = local[worker] - 20
    return balance


def main() -> None:
    slots = [worker for worker, count in STEPS.items() for _ in range(count)]
    finals = {run(order) for order in set(permutations(slots))}
    print(sum(finals))


if __name__ == "__main__":
    main()
