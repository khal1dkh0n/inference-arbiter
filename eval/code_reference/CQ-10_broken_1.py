"""Seeded bug 1 for CQ-10 (eval/code_questions.yaml): ties keep their order of first appearance.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

import re
from collections import Counter


def top_k_words(text: str, k: int) -> list[tuple[str, int]]:
    if k <= 0:
        return []
    counts = Counter(word.lower() for word in re.findall(r"[A-Za-z]+", text))
    ordered = sorted(counts.items(), key=lambda item: -item[1])
    return ordered[:k]
