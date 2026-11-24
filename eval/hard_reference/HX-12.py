"""HX-12: interchange for a 24-row multi-currency batch, BCD amounts, half-even rounding."""

from decimal import ROUND_HALF_EVEN, Decimal

QUESTION_DATA = [
    "01 CD X KWD 000000401831", "02 CC D JPY 000000286463", "03 CC X USD 000000200699",
    "04 CD D USD 000000031250", "05 CM X EUR 000000044270", "06 CC X JPY 000000222579",
    "07 CD X USD 000000096353", "08 CD D USD 000000217716", "09 CC D USD 000000146397",
    "10 CC X JPY 000000278832", "11 CC X EUR 000000007700", "12 CD X EUR 000000248392",
    "13 CM D KWD 000000332514", "14 CD D EUR 000000040584", "15 CD D EUR 000000086244",
    "16 CM D USD 000000074277", "17 CC X JPY 000000124050", "18 CM D JPY 000000002600",
    "19 CC D USD 000000009005", "20 CD D KWD 000000387185", "21 CD D JPY 000000043734",
    "22 CM D JPY 000000095787", "23 CM X USD 000000045832", "24 CM D KWD 000000816347",
]

RATE_PERCENT = {
    ("CD", "D"): "0.20", ("CD", "X"): "1.15", ("CC", "D"): "0.30",
    ("CC", "X"): "1.50", ("CM", "D"): "1.75", ("CM", "X"): "2.00",
}
EXPONENT = {"USD": 2, "EUR": 2, "JPY": 0, "KWD": 3}
EUR_PER_UNIT = {"USD": "0.92", "EUR": "1", "JPY": "0.0061", "KWD": "2.98"}


def main() -> None:
    total_eur = Decimal(0)
    for row in QUESTION_DATA:
        _number, card_type, region, currency, bcd_amount = row.split()
        amount_minor = int(bcd_amount)
        fee_minor = (amount_minor * Decimal(RATE_PERCENT[(card_type, region)]) / 100).quantize(
            Decimal(1), rounding=ROUND_HALF_EVEN
        )
        fee_major = fee_minor / (Decimal(10) ** EXPONENT[currency])
        total_eur += fee_major * Decimal(EUR_PER_UNIT[currency])
    total_cents = (total_eur * 100).quantize(Decimal(1), rounding=ROUND_HALF_EVEN)
    print(total_cents)


if __name__ == "__main__":
    main()
