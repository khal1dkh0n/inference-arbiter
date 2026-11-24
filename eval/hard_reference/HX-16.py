"""HX-16: HMAC-SHA256 webhook signature; first four bytes as an unsigned integer."""

import hashlib
import hmac

QUESTION_DATA = [
    "test-signing-key-0001",
    "1795434000",
    '{"id":"evt_test_0042","type":"payment.captured","amount":18750,"currency":"EUR"}',
]


def main() -> None:
    key, timestamp, body = QUESTION_DATA
    signed = f"{timestamp}.{body}".encode("utf-8")
    digest = hmac.new(key.encode("utf-8"), signed, hashlib.sha256).digest()
    print(int.from_bytes(digest[:4], "big"))


if __name__ == "__main__":
    main()
