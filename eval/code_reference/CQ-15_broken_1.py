"""Seeded bug 1 for CQ-15 (eval/code_questions.yaml): the groups are read most significant first.

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
        value = (value << 7) | (byte & 0x7F)
        if byte & 0x80 == 0:
            return value, position
    raise ValueError("varint is longer than 10 bytes")
