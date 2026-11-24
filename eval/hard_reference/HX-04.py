"""HX-04: parse an ASCII ISO 8583 message (hex bitmaps, LLVAR fields) and return DE 41."""

QUESTION_DATA = [
    "0200F238448128C080000000000014000000164761739001010119003000000000018750112214300500412715300511225411"
    "051000840012877314761739001010119D29122011234567632614004127414001280004140012877009781141400128774100041400128"
]

# data element -> fixed length, or "LL" for a two-digit length prefix
FIELD_SPEC = {
    2: "LL", 3: 6, 4: 12, 7: 10, 11: 6, 12: 6, 13: 4, 18: 4, 22: 3, 25: 2, 32: "LL", 35: "LL",
    37: 12, 41: 8, 42: 15, 49: 3, 100: "LL", 102: "LL",
}


def parse(message: str) -> dict[int, str]:
    position = 4
    bitmap = int(message[position : position + 16], 16)
    position += 16
    bit_count = 64
    if bitmap >> 63:
        bitmap = (bitmap << 64) | int(message[position : position + 16], 16)
        position += 16
        bit_count = 128
    present = [number for number in range(2, bit_count + 1) if (bitmap >> (bit_count - number)) & 1]
    fields: dict[int, str] = {}
    for number in present:
        spec = FIELD_SPEC[number]
        if spec == "LL":
            length = int(message[position : position + 2])
            position += 2
        else:
            length = spec
        fields[number] = message[position : position + length]
        position += length
    assert position == len(message)
    return fields


def main() -> None:
    print(parse(QUESTION_DATA[0])[41])


if __name__ == "__main__":
    main()
