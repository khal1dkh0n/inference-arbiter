"""HX-07: CRC-16 (poly 0x1021, init 0xFFFF, MSB first, no reflection, no final XOR), as decimal."""

QUESTION_DATA = ["9F02060000000187505F2A020978"]


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def main() -> None:
    assert crc16(b"123456789") == 0x29B1  # the standard check value for these parameters
    print(crc16(bytes.fromhex(QUESTION_DATA[0])))


if __name__ == "__main__":
    main()
