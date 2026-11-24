"""HR-09: which of five register histories are linearizable, by exhaustive search."""

QUESTION_DATA = [
    "C3 read -> 0   1-7", "C1 read -> 0   2-3", "C1 read -> 3   4-10", "C2 write(1)    5-6",
    "C3 write(2)    8-12", "C2 write(3)    9-14", "C1 read -> 2   11-13",
    "C1 read -> 0   1-2", "C3 read -> 1   3-5", "C2 write(1)    4-10", "C1 write(2)    6-7",
    "C3 read -> 2   8-13", "C1 write(3)    9-11", "C2 read -> 3   12-14",
    "C1 read -> 0   1-3", "C3 read -> 3   2-11", "C2 write(1)    4-7", "C1 write(2)    5-6",
    "C2 write(3)    8-12", "C1 read -> 2   9-10", "C3 read -> 2   13-14",
    "C1 write(1)    1-10", "C2 write(2)    2-11", "C3 read -> 2   3-4", "C3 read -> 1   5-6",
    "C3 read -> 2   7-8", "C1 read -> 2   12-13",
    "C3 read -> 1   7-8", "C1 read -> 1   12-13",
]

HISTORIES = [QUESTION_DATA[0:7], QUESTION_DATA[7:14], QUESTION_DATA[14:21], QUESTION_DATA[21:27],
             QUESTION_DATA[21:25] + QUESTION_DATA[27:29]]


def parse(line: str) -> tuple[str, int, int, int]:
    """(kind, value, invoked, returned)."""
    parts = line.split()
    invoked, returned = (int(number) for number in parts[-1].split("-"))
    if parts[1] == "read":
        return "read", int(parts[3]), invoked, returned
    return "write", int(parts[1][len("write("):-1]), invoked, returned


def linearizable(operations: list[tuple[str, int, int, int]]) -> bool:
    def search(remaining: frozenset[int], value: int) -> bool:
        if not remaining:
            return True
        first_return = min(operations[index][3] for index in remaining)
        for index in remaining:
            kind, operand, invoked, _returned = operations[index]
            if invoked > first_return:
                continue  # another pending operation returned before this one was invoked
            if kind == "write" and search(remaining - {index}, operand):
                return True
            if kind == "read" and operand == value and search(remaining - {index}, value):
                return True
        return False

    return search(frozenset(range(len(operations))), 0)


def main() -> None:
    print("".join("Y" if linearizable([parse(line) for line in history]) else "N" for history in HISTORIES))


if __name__ == "__main__":
    main()
