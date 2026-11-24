"""Seeded bug 2 for CQ-15 (eval/code_questions.yaml): the byte count is returned, not the next offset.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

def decode_varint(data: bytes, offset: int = 0) -> tuple[int, int]:
    value = 0
    position = offset
    for group in range(10):
        if position >= len(data):
            raise ValueError("data ends inside a varint")
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << (7 * group)
        if byte & 0x80 == 0:
            return value, position - offset
    raise ValueError("varint is longer than 10 bytes")
