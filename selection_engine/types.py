from __future__ import annotations

"""
types.py
Core dataclasses shared by every stage of the selection engine.

Ranking is percentile-based (see MarketProfile.rank_signal_fn /
historical_percentiles), not a weighted 0-100 composite score. Category
scorers still exist for diagnostics (form, matchup, market_compatibility,
stability) but do not decide ranking or BET eligibility — see
markets/tennis_set1_over_7_5.py.
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

PERCENTILE_KEYS: Tuple[str, ...] = ("p0", "p20", "p40", "p60", "p80", "p100")


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
    """A single 0-20 diagnostic category score plus the notes that justify it.

    Diagnostic only — does not feed ranking or BET eligibility.
    """

    value: float
    notes: List[str] = field(default_factory=list)


# Market-specific hooks. Each hard filter / veto returns None (pass) or a
# short machine-readable reason string (fail). Each category scorer returns
# a diagnostic CategoryScore for the given match. rank_signal_fn returns the
# raw ranking value (e.g. a calibrated probability like p_cal_adj) — not
# rescaled to any 0-20/0-100 range — or None if no signal is available.
HardFilterFn = Callable[[MatchInput], Optional[str]]
VetoFn = Callable[[MatchInput], Optional[str]]
CategoryScorerFn = Callable[[MatchInput], CategoryScore]
RankSignalFn = Callable[[MatchInput], Optional[float]]


@dataclass
class MarketProfile:
    """Everything the engine needs to evaluate matches for one specific market.

    Ranking/eligibility model: rank_signal_fn produces one raw value per
    match (e.g. p_cal_adj). historical_percentiles (p0/p20/p40/p60/p80/p100),
    computed from real historical outcomes for this exact market — never
    guessed — classify that value into a label and a bet_eligible flag:
    historical_percentile (0-100) >= bet_threshold_percentile (default 80,
    i.e. the historical top quintile) -> "TOP_HISTORICAL_QUINTILE",
    bet_eligible=True; 60-80 -> "HIGH"; 40-60 -> "MEDIUM"; 20-40 -> "LOW";
    below 20 -> "VERY_LOW". Only TOP_HISTORICAL_QUINTILE candidates are
    eligible for top_picks — if none qualify, the result is NO_BET regardless
    of how many matches survived filtering. historical_percentiles must be
    computed from the market's POST_HARD_FILTER / PRE_VETO population (veto
    is a selection filter, not part of the statistical universe rank_value is
    measured against) — see markets/tennis_set1_over_7_5.py.

    category_scorers (form, matchup, market_compatibility, stability) are
    still computed and attached to each result for diagnostics/logging — they
    never affect ranking or eligibility.
    """

    market_id: str
    sport: str
    top_n: int = 2

    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    min_sample_size: int = 0
    max_data_age_days: Optional[int] = None

    hard_filters: List[HardFilterFn] = field(default_factory=list)
    vetoes: List[VetoFn] = field(default_factory=list)
    category_scorers: Dict[str, CategoryScorerFn] = field(default_factory=dict)

    rank_signal_fn: Optional[RankSignalFn] = None
    historical_percentiles: Dict[str, float] = field(default_factory=dict)
    bet_threshold_percentile: float = 80.0


@dataclass
class MatchResult:
    """The fully evaluated outcome for one match."""

    match_id: str
    competitors: Tuple[str, str]
    status: str  # "QUALIFIED" | "ELIMINATED"
    elimination_reason: Optional[str] = None
    category_scores: Dict[str, CategoryScore] = field(default_factory=dict)
    rank_value: Optional[float] = None
    historical_percentile: Optional[float] = None
    label: str = ""  # "TOP_HISTORICAL_QUINTILE" | "HIGH" | "MEDIUM" | "LOW" | "VERY_LOW" | ""
    bet_eligible: bool = False
    data_quality: float = 1.0

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
    historical_p80: Optional[float] = None
