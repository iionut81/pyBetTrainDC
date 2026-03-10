from __future__ import annotations

from typing import Dict, Optional, Set

from config import CFG

_VAR = CFG["variance"]


def implied_probability(odds: Optional[float]) -> Optional[float]:
    if odds is None or odds <= 0:
        return None
    return 1.0 / odds


def classify_variance(var_value: float) -> str:
    if var_value <= _VAR["low"]:
        return "LOW"
    if var_value <= _VAR["low_medium"]:
        return "LOW-MEDIUM"
    if var_value <= _VAR["medium"]:
        return "MEDIUM"
    return "HIGH"


def evaluate_market(
    market: str,
    model_probability: float,
    variance_value: float,
    upset_frequency: float,
    offered_odds: Optional[float],
    min_dc_probability: float = 0.78,
    min_odds: float = 1.25,
    max_odds: float = 1.35,
    allowed_variance_classes: Optional[Set[str]] = None,
) -> Dict[str, object]:
    fair_odds = (1.0 / model_probability) if model_probability > 0 else None
    effective_odds = offered_odds
    odds_source = "market" if offered_odds is not None else "missing"

    imp = implied_probability(offered_odds)
    edge = None if imp is None else (model_probability - imp)
    variance_class = classify_variance(variance_value)

    pass_odds = offered_odds is not None and (min_odds <= offered_odds <= max_odds)
    pass_prob = model_probability >= min_dc_probability
    allowed = allowed_variance_classes if allowed_variance_classes is not None else {"LOW", "MEDIUM"}
    pass_variance = variance_class in allowed
    pass_edge = edge is not None and edge > 0.0

    recommended = bool(pass_odds and pass_prob and pass_variance and pass_edge)
    return {
        "market": market,
        "model_probability": model_probability,
        "offered_odds": offered_odds,
        "fair_odds": fair_odds,
        "effective_odds": effective_odds,
        "odds_source": odds_source,
        "implied_probability": imp,
        "edge": edge,
        "variance": variance_value,
        "variance_class": variance_class,
        "upset_frequency": upset_frequency,
        "recommended": recommended,
    }
