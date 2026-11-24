"""HX-08: XOR LRC over a framed message: every byte after STX up to and including ETX."""

from functools import reduce

QUESTION_DATA = [
    "02303230307C343736313733393030313031303131397C3030303030303031383735307C3937387CB9BCD408CC6C03"
]


def main() -> None:
    frame = bytes.fromhex(QUESTION_DATA[0])
    assert frame[0] == 0x02 and frame[-1] == 0x03
    print(reduce(lambda accumulator, byte: accumulator ^ byte, frame[1:], 0))


if __name__ == "__main__":
    main()
