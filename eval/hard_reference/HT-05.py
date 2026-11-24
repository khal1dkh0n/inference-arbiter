"""HT-05: strict BER-TLV walk; a child that overruns its template makes the message malformed."""

QUESTION_DATA = [
    "772B9F2608A1B2C3D4E5F60718BF0C085F2A0209789F02060000000249909F3602003C9F100706010A03A00000",
]


class Malformed(Exception):
    pass


def walk(data: bytes, found: dict[str, bytes]) -> None:
    position = 0
    while position < len(data):
        start = position
        position += 1
        if data[start] & 0x1F == 0x1F:
            while True:
                if position >= len(data):
                    raise Malformed("tag runs past the end")
                position += 1
                if not data[position - 1] & 0x80:
                    break
        tag = data[start:position]
        if position >= len(data):
            raise Malformed("no length byte")
        length = data[position]
        position += 1
        if length == 0x81:
            length = data[position]
            position += 1
        elif length >= 0x80:
            raise Malformed("unsupported length form")
        value = data[position:position + length]
        if len(value) < length:
            raise Malformed(f"{tag.hex().upper()} runs past the end of its template")
        position += length
        if tag[0] & 0x20:
            walk(value, found)
        else:
            found[tag.hex().upper()] = value


def main() -> None:
    message = bytes.fromhex(QUESTION_DATA[0])
    assert bytes.fromhex("9F0206000000024990") in message  # a plain search does find an amount
    try:
        found: dict[str, bytes] = {}
        walk(message, found)
        print(int(found["9F02"].hex()))
    except Malformed:
        print("MALFORMED")


if __name__ == "__main__":
    main()
