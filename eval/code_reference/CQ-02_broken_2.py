"""Seeded bug 2 for CQ-02 (eval/code_questions.yaml): a two-byte length is read little-endian.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

def parse_tlv(data: bytes) -> list[tuple[str, bytes]]:
    objects = []
    position = 0
    while position < len(data):
        if data[position] == 0x00:
            position += 1
            continue
        tag_start = position
        first_byte = data[position]
        position += 1
        if first_byte & 0x1F == 0x1F:
            while True:
                if position >= len(data):
                    raise ValueError("data ends inside a tag")
                tag_byte = data[position]
                position += 1
                if tag_byte & 0x80 == 0:
                    break
        tag = data[tag_start:position].hex().upper()
        if position >= len(data):
            raise ValueError("data ends before the length")
        length_byte = data[position]
        position += 1
        if length_byte < 0x80:
            length = length_byte
        elif length_byte in (0x81, 0x82):
            size = length_byte - 0x80
            if position + size > len(data):
                raise ValueError("data ends inside the length")
            length = int.from_bytes(data[position:position + size], "little")
            position += size
        else:
            raise ValueError("unsupported length form")
        if position + length > len(data):
            raise ValueError("data ends inside a value")
        objects.append((tag, data[position:position + length]))
        position += length
    return objects
