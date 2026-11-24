"""Seeded bug 2 for CQ-04 (eval/code_questions.yaml): the wrong polynomial.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

def crc16(data: bytes) -> int:
    register = 0xFFFF
    for byte in data:
        register ^= byte << 8
        for _ in range(8):
            if register & 0x8000:
                register = ((register << 1) ^ 0x8005) & 0xFFFF
            else:
                register = (register << 1) & 0xFFFF
    return register
