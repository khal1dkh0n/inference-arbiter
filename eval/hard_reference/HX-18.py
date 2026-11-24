"""HX-18: base64 then zlib; sum one column of the CSV inside for one currency."""

import base64
import zlib

QUESTION_DATA = [
    "eNo1j7sOwjAMAPd8S4bY8XOHMQuIGaGqEgxtpQoG/h43EaNP57O8zPv0fKzvPH32fV6nb34s22d935fXuu2pEarl8+2SCUkKBFDFDgSdWVNDgmGgALmnpqiSb9dTAKpcUnND6iB89ljxAsNwxhICMZeRgAJHUyuPplUEkNQqOXWgIAwcwCv2BBn50RBm6MAckcNwkWGYusdsYv8EoMUnID6O1IpK+gNv7T8H"
]


def main() -> None:
    text = zlib.decompress(base64.b64decode(QUESTION_DATA[0])).decode("ascii")
    header, *rows = text.splitlines()
    assert header == "merchant,currency,amount_minor"
    print(sum(int(amount) for _merchant, currency, amount in (row.split(",") for row in rows) if currency == "EUR"))


if __name__ == "__main__":
    main()
