"""HT-06: what the Java split method returns for a refund (integer division truncates toward zero)."""

QUESTION_DATA = ["-10000", "{1, 1, 1}", "-3334, -3333, -3333", "totalMinor * weights[i] / sum"]


def java_divide(dividend: int, divisor: int) -> int:
    quotient = abs(dividend) // abs(divisor)
    return quotient if (dividend < 0) == (divisor < 0) else -quotient


def split(total_minor: int, weights: list[int]) -> list[int]:
    total_weight = sum(weights)
    parts = [java_divide(total_minor * weight, total_weight) for weight in weights]
    remainder = total_minor - sum(parts)
    index = 0
    while index < remainder:  # for (int i = 0; i < remainder; i++)
        parts[index] += 1
        index += 1
    return parts


def main() -> None:
    assert sum(split(10000, [1, 1, 1])) == 10000 and split(10000, [1, 1, 1]) == [3334, 3333, 3333]
    parts = split(int(QUESTION_DATA[0]), [1, 1, 1])
    print(" ".join(str(part) for part in parts), f"sum={sum(parts)}")


if __name__ == "__main__":
    main()
