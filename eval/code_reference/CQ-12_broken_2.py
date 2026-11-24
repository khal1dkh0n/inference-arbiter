"""Seeded bug 2 for CQ-12 (eval/code_questions.yaml): an empty relative result is the empty string.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

def normalize_path(path: str) -> str:
    absolute = path.startswith("/")
    kept: list[str] = []
    for segment in path.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if kept and kept[-1] != "..":
                kept.pop()
            elif not absolute:
                kept.append("..")
            continue
        kept.append(segment)
    if absolute:
        return "/" + "/".join(kept)
    return "/".join(kept)
