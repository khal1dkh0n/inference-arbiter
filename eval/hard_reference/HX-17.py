"""HX-17: SHA-1 over the values of three tags picked out of a TLV record."""

import hashlib

QUESTION_DATA = ["701E5A0847617390010101195F24032912315F3401019F0702FF005F28020826"]
HASHED_TAGS = ["5A", "5F24", "9F07"]


def parse(data: bytes) -> list[tuple[str, bytes]]:
    found = []
    position = 0
    while position < len(data):
        start = position
        first_byte = data[position]
        position += 1
        if first_byte & 0x1F == 0x1F:
            position += 1
        tag = data[start:position].hex().upper()
        length = data[position]
        position += 1
        value = data[position : position + length]
        position += length
        if first_byte & 0x20:
            found.extend(parse(value))
        else:
            found.append((tag, value))
    return found


def main() -> None:
    objects = parse(bytes.fromhex(QUESTION_DATA[0]))
    hashed = b"".join(value for tag, value in objects if tag in HASHED_TAGS)
    assert len(hashed) == 13
    print(int.from_bytes(hashlib.sha1(hashed).digest()[:4], "big"))


if __name__ == "__main__":
    main()
