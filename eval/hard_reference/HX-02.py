"""HX-02: total number of value bytes held by primitive objects, at every nesting level."""

import importlib.util
from pathlib import Path

QUESTION_DATA = [
    "E181539F1A020826008407A0000000041010A58132500544454249548701010000BF0C8200129F4D020B0A004203465F20"
    "DF8129039F0A049F380C9F66049F02069F37045F2A025F2D04656E6465DF813004A5038701"
]


def load_parser():
    spec = importlib.util.spec_from_file_location("hx01", Path(__file__).with_name("HX-01.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse


def main() -> None:
    objects = load_parser()(bytes.fromhex(QUESTION_DATA[0]))
    print(sum(len(value) for _tag, value in objects))


if __name__ == "__main__":
    main()
