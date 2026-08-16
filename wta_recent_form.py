from __future__ import annotations

"""
wta_recent_form.py
Recent-form signal for selection_engine's TENNIS_SET1_OVER_7_5 market: how
erratic a player's service performance has been over her last N matches (any
surface), computed from data/historical/wta_matches_combined.csv — the same
Sackmann-based dataset used to train Elo/Glicko/Markov ratings elsewhere in
this project. No lookahead: only matches strictly before `as_of` are used.

Real WTA per-match serve-points-won% has a much tighter raw spread than the
0-1 scale selection_engine.markets.tennis_set1_over_7_5 expects for
`recent_form_variance_a/b`. RAW_FLOOR/RAW_CEILING renormalize the observed
raw range onto that 0-1 scale so the category scorer's thresholds (consistent
<=0.15, erratic >=0.40) are meaningful on real data. These two constants are
the 5th/95th percentile of the 12-match rolling std across every player with
>=12 historical matches (946 players sampled 2026-08-16, median 0.083,
5th pct 0.050, 95th pct 0.125) — i.e. "erraticism relative to the rest of the
tour", not an absolute bound. Revisit if the historical dataset changes
materially.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_N_MATCHES = 12
RAW_FLOOR = 0.050
RAW_CEILING = 0.125

_HISTORY_PATH = Path(__file__).resolve().parent / "data" / "historical" / "wta_matches_combined.csv"

_WINNER_COLS = {"winner_name": "player", "w_svpt": "svpt", "w_1stWon": "first_won", "w_2ndWon": "second_won"}
_LOSER_COLS = {"loser_name": "player", "l_svpt": "svpt", "l_1stWon": "first_won", "l_2ndWon": "second_won"}


def _long_format(history: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, match) with a serve-points-won% performance metric."""
    winners = history[list(_WINNER_COLS)].rename(columns=_WINNER_COLS)
    losers = history[list(_LOSER_COLS)].rename(columns=_LOSER_COLS)
    combined_dates = pd.concat([history["match_date"], history["match_date"]], ignore_index=True)

    long_df = pd.concat([winners, losers], ignore_index=True)
    long_df["match_date"] = pd.to_datetime(combined_dates, errors="coerce")
    long_df = long_df.dropna(subset=["match_date", "svpt"])
    long_df = long_df[long_df["svpt"] > 0]
    long_df["serve_pts_won_pct"] = (long_df["first_won"] + long_df["second_won"]) / long_df["svpt"]
    return long_df


def load_history(path: Optional[Path] = None) -> pd.DataFrame:
    raw = pd.read_csv(path or _HISTORY_PATH, low_memory=False)
    return _long_format(raw)


def _normalize(raw_std: float) -> float:
    span = RAW_CEILING - RAW_FLOOR
    return max(0.0, min(1.0, (raw_std - RAW_FLOOR) / span))


def recent_form_variance(
    history_long: pd.DataFrame,
    player: str,
    as_of: pd.Timestamp,
    n_matches: int = DEFAULT_N_MATCHES,
) -> Optional[float]:
    """Normalized 0-1 erraticism index over a player's last n_matches strictly
    before `as_of`. Returns None (never estimated) if fewer than n_matches
    of history are available for this player."""
    rows = history_long[(history_long["player"] == player) & (history_long["match_date"] < as_of)]
    rows = rows.sort_values("match_date", ascending=False).head(n_matches)
    if len(rows) < n_matches:
        return None
    raw_std = float(rows["serve_pts_won_pct"].std(ddof=0))
    return _normalize(raw_std)


PlayerIndex = Dict[str, Tuple[np.ndarray, np.ndarray]]


def build_player_index(history_long: pd.DataFrame) -> PlayerIndex:
    """Pre-sort each player's (date, serve_pts_won_pct) once so repeated
    recent_form_variance_indexed() calls (e.g. over thousands of backtest
    rows) are O(log n) lookups instead of O(n) scans of the whole history."""
    index: PlayerIndex = {}
    for player, group in history_long.groupby("player", sort=False):
        g = group.sort_values("match_date")
        index[player] = (g["match_date"].to_numpy(), g["serve_pts_won_pct"].to_numpy())
    return index


def recent_form_variance_indexed(
    index: PlayerIndex,
    player: str,
    as_of: pd.Timestamp,
    n_matches: int = DEFAULT_N_MATCHES,
) -> Optional[float]:
    """Same semantics as recent_form_variance(), backed by build_player_index()."""
    data = index.get(player)
    if data is None:
        return None
    dates, pcts = data
    pos = int(np.searchsorted(dates, np.datetime64(as_of), side="left"))
    if pos < n_matches:
        return None
    window = pcts[pos - n_matches : pos]
    raw_std = float(window.std(ddof=0))
    return _normalize(raw_std)
