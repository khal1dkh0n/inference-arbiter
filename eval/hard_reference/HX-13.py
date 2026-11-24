"""HX-13: what a TLV tag lister with two bugs returns for a real input."""

QUESTION_DATA = [
    "7082002E9F4681039F02065A0847617390010101195F3401019F0702FF008E0A00000000000000001F039F0D05B450840000"
]


def tags(data):
    out = []
    i = 0
    while i < len(data):
        tag = data[i]
        i += 1
        if tag & 0x1F == 0x1F:
            tag = (tag << 8) | data[i]
            i += 1
        length = data[i]
        i += 1
        if length & 0x80:
            length = data[i]
            i += 1
        out.append(tag)
        if not tag & 0x20:
            i += length
    return out


def main() -> None:
    print(sum(tags(bytes.fromhex(QUESTION_DATA[0]))))


if __name__ == "__main__":
    main()
