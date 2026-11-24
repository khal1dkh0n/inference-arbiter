"""HD-03: build the TVR from the stated facts, then run terminal action analysis."""

from datetime import date

QUESTION_DATA = [
    "008C", "008D", "2027-04-01", "2029-03-31", "2027-03-31", "80.00", "20.00", "100.00", "0113", "010E",
    "00 10 40 00 00", "04 00 00 00 00", "F0 40 5C 40 00", "0C 08 00 B0 00", "F0 40 5C 40 00", "0C 08 20 30 00",
]


def as_bytes(text: str) -> bytes:
    return bytes.fromhex(text.replace(" ", ""))


def hits(tvr: bytes, issuer_code: str, terminal_code: str) -> bool:
    return any(t & (i | c) for t, i, c in zip(tvr, as_bytes(issuer_code), as_bytes(terminal_code)))


def main() -> None:
    card_version, terminal_version, effective, expiry, today = QUESTION_DATA[0:5]
    amount, logged, floor_limit = (round(float(value) * 100) for value in QUESTION_DATA[5:8])
    atc, last_online_atc = (int(value, 16) for value in QUESTION_DATA[8:10])
    lower_limit, upper_limit = 5, 8
    tvr = bytearray(5)
    # Byte 1: static data authentication was performed and succeeded, so nothing is set.
    if card_version != terminal_version:
        tvr[1] |= 0x80
    if date.fromisoformat(today) > date.fromisoformat(expiry):
        tvr[1] |= 0x40
    if date.fromisoformat(today) < date.fromisoformat(effective):
        tvr[1] |= 0x20
    if last_online_atc == 0:
        tvr[1] |= 0x08
    # Offline PIN was attempted with the try counter already at zero; the CVM rule does not allow a next rule.
    tvr[2] |= 0x20 | 0x80
    if amount + logged >= floor_limit:
        tvr[3] |= 0x80
    if atc - last_online_atc > lower_limit:
        tvr[3] |= 0x40
    if atc - last_online_atc > upper_limit:
        tvr[3] |= 0x20
    iac_denial, tac_denial, iac_online, tac_online, iac_default, tac_default = QUESTION_DATA[10:16]
    if hits(tvr, iac_denial, tac_denial):
        first, offline = "AAC", "AAC"
    elif hits(tvr, iac_online, tac_online):
        first = "ARQC"
        offline = "AAC" if hits(tvr, iac_default, tac_default) else "TC"
    else:
        first, offline = "TC", "TC"
    print(tvr.hex().upper(), first, offline)


if __name__ == "__main__":
    main()
