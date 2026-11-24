"""Seeded bug 2 for CQ-18 (eval/code_questions.yaml): the last piece of a long word is closed as its own line.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

def wrap_text(text: str, width: int) -> list[str]:
    if width < 1:
        raise ValueError("width must be at least 1")
    lines: list[str] = []
    current = ""
    for word in text.split():
        if len(word) > width:
            if current:
                lines.append(current)
            while len(word) > width:
                lines.append(word[:width])
                word = word[width:]
            lines.append(word)
            current = ""
            continue
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current = current + " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
