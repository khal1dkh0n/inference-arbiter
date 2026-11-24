"""HX-01: walk a nested BER-TLV structure and return tag 9F02 as a decimal integer."""

QUESTION_DATA = [
    "7781849F2701809F360201A79F4B3079C4121FC60857DFFEFC74B999CC85F1909F0206000000004200AC46C8BCDF728E70FE"
    "5F2A0208406CCE861E33E6B0EA9F10120110A040032200009F0206000000990000FFBF0C139F4D020B0A9F0206000000018750"
    "5F2A0209789F26086FE62AADA4D044EA57104761739001010119D291220112345678"
]


def parse(data: bytes) -> list[tuple[str, bytes]]:
    """Return (tag, value) for every primitive object at any depth, in order."""
    found: list[tuple[str, bytes]] = []
    position = 0
    while position < len(data):
        if data[position] == 0x00:
            position += 1
            continue
        tag_start = position
        first_byte = data[position]
        position += 1
        if first_byte & 0x1F == 0x1F:
            while data[position] & 0x80:
                position += 1
            position += 1
        tag = data[tag_start:position].hex().upper()
        length = data[position]
        position += 1
        if length & 0x80:
            length_bytes = length & 0x7F
            length = int.from_bytes(data[position : position + length_bytes], "big")
            position += length_bytes
        value = data[position : position + length]
        position += length
        if first_byte & 0x20:
            found.extend(parse(value))
        else:
            found.append((tag, value))
    return found


def main() -> None:
    objects = parse(bytes.fromhex(QUESTION_DATA[0]))
    amounts = [value for tag, value in objects if tag == "9F02"]
    assert len(amounts) == 1
    print(int(amounts[0].hex()))


if __name__ == "__main__":
    main()
