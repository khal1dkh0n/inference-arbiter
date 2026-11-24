"""HR-11: number of concurrent event pairs in a four-process message history."""

from itertools import combinations

QUESTION_DATA = [
    "P1: send m5 to P2, send m6 to P3, local, local, local",
    "P2: send m4 to P4, local, receive m5, local, local",
    "P3: receive m1, send m2 to P4, receive m3, receive m6",
    "P4: send m1 to P3, send m3 to P3, local, receive m4, receive m2",
]


def main() -> None:
    events: list[tuple[str, int]] = []
    sent: dict[str, tuple[str, int]] = {}
    received: dict[str, tuple[str, int]] = {}
    for line in QUESTION_DATA:
        process, listing = line.split(": ")
        for position, text in enumerate(listing.split(", ")):
            event = (process, position)
            events.append(event)
            words = text.split()
            if words[0] == "send":
                sent[words[1]] = event
            elif words[0] == "receive":
                received[words[1]] = event
    assert sent.keys() == received.keys()
    successors = {event: set() for event in events}
    for process, position in events:
        if position:
            successors[(process, position - 1)].add((process, position))
    for message, event in sent.items():
        successors[event].add(received[message])

    def reachable(start: tuple[str, int]) -> set[tuple[str, int]]:
        seen: set[tuple[str, int]] = set()
        stack = [start]
        while stack:
            for following in successors[stack.pop()]:
                if following not in seen:
                    seen.add(following)
                    stack.append(following)
        return seen

    after = {event: reachable(event) for event in events}
    print(sum(1 for a, b in combinations(events, 2) if b not in after[a] and a not in after[b]))


if __name__ == "__main__":
    main()
