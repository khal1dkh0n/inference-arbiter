"""Seeded bug 1 for CQ-13 (eval/code_questions.yaml): equal remainders go to the higher index.

It shows that the reference tests fail a wrong solution. It is never part of a prompt.
"""

def allocate(total: int, weights: list[int]) -> list[int]:
    if total < 0 or not weights or any(weight < 0 for weight in weights):
        raise ValueError("total and weights must not be negative, and weights must not be empty")
    weight_sum = sum(weights)
    if weight_sum == 0:
        raise ValueError("the weights add up to zero")
    shares = [total * weight // weight_sum for weight in weights]
    remainders = [total * weight % weight_sum for weight in weights]
    leftover = total - sum(shares)
    by_remainder = sorted(range(len(weights)), key=lambda index: (-remainders[index], -index))
    for index in by_remainder[:leftover]:
        shares[index] += 1
    return shares
