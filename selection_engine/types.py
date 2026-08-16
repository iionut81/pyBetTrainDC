from __future__ import annotations

"""
types.py
Core dataclasses shared by every stage of the selection engine.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

CATEGORY_NAMES: Tuple[str, ...] = (
    "form",
    "matchup",
    "statistics",
    "market_compatibility",
    "stability",
)
CATEGORY_MAX = 20.0


@dataclass
class MatchInput:
    """A single candidate match/event to be evaluated for one market."""

    match_id: str
    market: str
    sport: str
    competitors: Tuple[str, str]
    stats: Dict[str, Any] = field(default_factory=dict)
    odds: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryScore:
    """A single 0-20 category score plus the notes that justify it."""

    value: float
    notes: List[str] = field(default_factory=list)


# Market-specific hooks. Each hard filter / veto returns None (pass) or a
# short machine-readable reason string (fail). Each category scorer returns
# a CategoryScore for the given match.
HardFilterFn = Callable[[MatchInput], Optional[str]]
VetoFn = Callable[[MatchInput, Dict[str, "CategoryScore"]], Optional[str]]
CategoryScorerFn = Callable[[MatchInput], CategoryScore]
ScoreFn = Callable[[Dict[str, "CategoryScore"]], float]


@dataclass
class MarketProfile:
    """Everything the engine needs to evaluate matches for one specific market.

    All thresholds are plain fields with defaults — nothing is hardcoded in
    the engine itself. Weights are kept for documentation/future use; the
    actual per-category score is always capped at 0-20 (spec: fewer knobs,
    fixed 0-100 total).

    score_fn: optional override for how the 5 category scores become the
    0-100 ranking score. Default (None) sums all 5 categories and applies the
    contradiction penalty — the original composite-score design. A market can
    set score_fn to drive ranking off a single trusted category instead (e.g.
    a category built from the production model's own calibrated probability)
    once backtesting shows the composite score doesn't beat that signal alone
    — see markets/tennis_set1_over_7_5.py for a market that does this. The
    other categories still get computed and attached to the result as
    diagnostics; they just stop deciding the ranking. When score_fn is set,
    the contradiction check is skipped (no penalty applies).
    """

    market_id: str
    sport: str
    minimum_score: float = 80.0
    top_n: int = 2
    allow_no_bet: bool = True

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "form": 0.20,
            "matchup": 0.20,
            "statistics": 0.30,
            "market_compatibility": 0.20,
            "stability": 0.10,
        }
    )

    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    min_sample_size: int = 0
    max_data_age_days: Optional[int] = None

    contradiction_high: float = 17.0
    contradiction_low: float = 10.0
    contradiction_spread: float = 10.0
    contradiction_penalty: float = 8.0

    hard_filters: List[HardFilterFn] = field(default_factory=list)
    vetoes: List[VetoFn] = field(default_factory=list)
    category_scorers: Dict[str, CategoryScorerFn] = field(default_factory=dict)
    score_fn: Optional[ScoreFn] = None


@dataclass
class MatchResult:
    """The fully evaluated outcome for one match."""

    match_id: str
    competitors: Tuple[str, str]
    status: str  # "QUALIFIED" | "ELIMINATED"
    elimination_reason: Optional[str] = None
    category_scores: Dict[str, CategoryScore] = field(default_factory=dict)
    raw_total: float = 0.0
    contradiction: bool = False
    contradiction_penalty: float = 0.0
    contradiction_notes: List[str] = field(default_factory=list)
    final_score: float = 0.0
    data_quality: float = 1.0
    confidence: str = "LOW"

    @property
    def strengths(self) -> List[str]:
        notes: List[str] = []
        for cs in self.category_scores.values():
            notes.extend(n for n in cs.notes if n.startswith("+"))
        return notes

    @property
    def risks(self) -> List[str]:
        notes: List[str] = []
        for cs in self.category_scores.values():
            notes.extend(n for n in cs.notes if n.startswith("-"))
        if self.contradiction:
            notes.extend(n for n in self.contradiction_notes if n.startswith("-"))
        return notes


@dataclass
class EngineResult:
    """Output of a full engine run for one market over N matches."""

    market_id: str
    total_analyzed: int
    eliminated: List[MatchResult] = field(default_factory=list)
    qualified: List[MatchResult] = field(default_factory=list)
    top_picks: List[MatchResult] = field(default_factory=list)
    decision: str = "NO_BET"  # "BET" | "NO_BET"
