from __future__ import annotations

from typing import Iterable


def american_to_implied_prob(odds: int | float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def devig_two_way_probabilities(team_a_odds: int | float, team_b_odds: int | float) -> tuple[float, float]:
    raw_a = american_to_implied_prob(team_a_odds)
    raw_b = american_to_implied_prob(team_b_odds)
    total = raw_a + raw_b
    if total <= 0:
        raise ValueError("Implied probabilities must sum to a positive value.")
    return raw_a / total, raw_b / total


def consensus_probability(
    quotes: Iterable[dict],
    *,
    probability_key: str = "probability",
    weight_key: str = "weight",
) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    for quote in quotes:
        probability = float(quote[probability_key])
        weight = float(quote.get(weight_key, 1.0))
        total_weight += weight
        weighted_sum += probability * weight
    if total_weight <= 0:
        raise ValueError("Consensus requires positive total weight.")
    return weighted_sum / total_weight

