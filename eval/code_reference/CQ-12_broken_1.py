"""Seeded bug 1 for CQ-12 (eval/code_questions.yaml): a relative path loses its leading parents.

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
            continue
        kept.append(segment)
    if absolute:
        return "/" + "/".join(kept)
    return "/".join(kept) or "."
