"""HX-14: RSA recovery with public exponent 3, then a field of the recovered block."""

QUESTION_DATA = ["B2764CD65A5BE7B2050CB0D5439DC0CD", "A6F67D445A09812EF39CCF5A55E7639D"]


def main() -> None:
    modulus, signed = (int(value, 16) for value in QUESTION_DATA)
    recovered = pow(signed, 3, modulus).to_bytes(16, "big")
    assert recovered[0] == 0x6A and recovered[-1] == 0xBC
    print(recovered[2:6].hex())


if __name__ == "__main__":
    main()
