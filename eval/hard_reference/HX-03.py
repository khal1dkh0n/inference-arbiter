"""HX-03: build the data field a DOL asks for, with EMV padding and truncation rules."""

QUESTION_DATA = [
    "9F02049F03069F1A0395055F2A029A039C019F37045F200A9F4C049F35019F21029F1E04",
    "000000018750", "000000000000", "0826", "0080008000", "0978", "271122", "3C91A7E2",
    "444F452F4A", "143005", "54524D3030303432",
]

# tag -> (format, value hex). 9F4C is requested by the DOL but the terminal has no value for it.
TERMINAL_DATA = {
    "9F02": ("n", "000000018750"),
    "9F03": ("n", "000000000000"),
    "9F1A": ("n", "0826"),
    "95": ("b", "0080008000"),
    "5F2A": ("n", "0978"),
    "9A": ("n", "271122"),
    "9C": ("n", "00"),
    "9F37": ("b", "3C91A7E2"),
    "5F20": ("an", "444F452F4A"),
    "9F35": ("n", "22"),
    "9F21": ("n", "143005"),
    "9F1E": ("an", "54524D3030303432"),
}


def parse_dol(dol: bytes) -> list[tuple[str, int]]:
    entries = []
    position = 0
    while position < len(dol):
        start = position
        first_byte = dol[position]
        position += 1
        if first_byte & 0x1F == 0x1F:
            while dol[position] & 0x80:
                position += 1
            position += 1
        entries.append((dol[start:position].hex().upper(), dol[position]))
        position += 1
    return entries


def fit(data_format: str, value: bytes, wanted: int) -> bytes:
    if len(value) > wanted:
        return value[-wanted:] if data_format == "n" else value[:wanted]
    padding = bytes(wanted - len(value))
    return padding + value if data_format == "n" else value + padding


def main() -> None:
    output = b""
    for tag, wanted in parse_dol(bytes.fromhex(QUESTION_DATA[0])):
        if tag not in TERMINAL_DATA:
            output += bytes(wanted)
            continue
        data_format, value_hex = TERMINAL_DATA[tag]
        output += fit(data_format, bytes.fromhex(value_hex), wanted)
    print(output.hex().upper())


if __name__ == "__main__":
    main()
