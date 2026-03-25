from __future__ import annotations

"""
wta_ratings.py
Compute rolling surface-specific player strength ratings from historical WTA data.
Optimized: pre-builds a player-perspective stats table once, then lookups are O(1).
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Mapping

import numpy as np
import pandas as pd

from wta_markov import PlayerServeStats

STAT_COLS = [
    "1stServeIn_pct", "1stServeWon_pct", "2ndServeWon_pct",
    "aceRate", "bpSaved_pct", "returnPtsWon_pct", "bpConverted_pct",
]


@dataclass
class PlayerRating:
    player_id: int
    player_name: str
    surface: str
    stats: PlayerServeStats
    last_match_date: pd.Timestamp
    serve_strength: float = 0.0
    return_strength: float = 0.0
    pressure_rating: float = 0.0


def build_player_match_stats(
    matches: pd.DataFrame,
    tier_weights: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Build a table where each row is one player's stats from one match.

    This is the key optimization: we do this ONCE and then filter by player/surface/date.

    If ``tier_weights`` is set and ``tourney_level`` exists on ``matches``, adds column
    ``tier_weight`` per row (Sackmann codes e.g. G, P, I, F). Use ``__default__`` for unknown levels.
    """
    # Winner perspective
    w = matches[["match_date", "surface", "winner_id",
                  "w_1stServeIn_pct", "w_1stServeWon_pct", "w_2ndServeWon_pct",
                  "w_aceRate", "w_bpSaved_pct", "w_returnPtsWon_pct", "w_bpConverted_pct"]].copy()
    w.columns = ["match_date", "surface", "player_id"] + STAT_COLS

    # Loser perspective
    l = matches[["match_date", "surface", "loser_id",
                  "l_1stServeIn_pct", "l_1stServeWon_pct", "l_2ndServeWon_pct",
                  "l_aceRate", "l_bpSaved_pct", "l_returnPtsWon_pct", "l_bpConverted_pct"]].copy()
    l.columns = ["match_date", "surface", "player_id"] + STAT_COLS

    if tier_weights and "tourney_level" in matches.columns:
        default_w = float(tier_weights.get("__default__", 1.0))

        def _tw(level: object) -> float:
            if level is None or (isinstance(level, float) and pd.isna(level)):
                return default_w
            key = str(level).strip()
            return float(tier_weights.get(key, default_w))

        tl = matches["tourney_level"]
        w["tier_weight"] = tl.map(_tw).astype(float)
        l["tier_weight"] = tl.map(_tw).astype(float)

    pms = pd.concat([w, l], ignore_index=True)
    pms["player_id"] = pms["player_id"].astype(int)
    pms = pms.sort_values("match_date").reset_index(drop=True)
    return pms


def _exponential_weights(n: int, decay: float) -> np.ndarray:
    w = np.exp(-decay * np.arange(n))
    return w / w.sum()


def compute_player_stats_fast(
    pms: pd.DataFrame,
    player_id: int,
    surface: str,
    window: int = 20,
    decay: float = 0.05,
    reference_date: Optional[pd.Timestamp] = None,
) -> Optional[PlayerServeStats]:
    """Fast lookup from pre-built player match stats table."""
    mask = (pms["player_id"] == player_id) & (pms["surface"] == surface)
    if reference_date is not None:
        mask = mask & (pms["match_date"] < reference_date)

    subset = pms.loc[mask]
    if len(subset) == 0:
        return None

    # Take last `window` matches (already sorted by date)
    subset = subset.tail(window)
    # Reverse so most recent is first for weighting
    vals = subset[STAT_COLS].values[::-1]  # shape (n, 7)
    n = len(vals)

    if n < 3:
        return None

    weights = _exponential_weights(n, decay)
    if "tier_weight" in subset.columns:
        tw = subset["tier_weight"].to_numpy(dtype=float)[::-1]
        weights = weights * tw
        s = float(weights.sum())
        if s > 1e-12:
            weights = weights / s

    def wmean(col_idx: int) -> float:
        v = vals[:, col_idx].astype(float)
        valid = ~np.isnan(v)
        if valid.sum() == 0:
            return 0.5
        w = weights[valid]
        return float(np.average(v[valid], weights=w / w.sum()))

    return PlayerServeStats(
        first_serve_in_pct=wmean(0),
        first_serve_won_pct=wmean(1),
        second_serve_won_pct=wmean(2),
        ace_rate=wmean(3),
        bp_saved_pct=wmean(4),
        return_pts_won_pct=wmean(5),
        bp_converted_pct=wmean(6),
        n_matches=n,
    )


# Keep old name as alias for backward compat in train_wta.py
compute_player_stats = compute_player_stats_fast


def build_all_ratings(
    matches: pd.DataFrame,
    reference_date: pd.Timestamp,
    surfaces: List[str],
    window: int = 20,
    decay: float = 0.05,
    min_matches: int = 10,
    active_days: int = 365,
) -> Dict[Tuple[int, str], PlayerRating]:
    cutoff = reference_date - pd.Timedelta(days=active_days)
    recent = matches[matches["match_date"] >= cutoff]
    player_ids = set(recent["winner_id"].unique()) | set(recent["loser_id"].unique())

    names: Dict[int, str] = {}
    for _, r in matches[["winner_id", "winner_name"]].drop_duplicates("winner_id").iterrows():
        names[int(r["winner_id"])] = str(r["winner_name"])
    for _, r in matches[["loser_id", "loser_name"]].drop_duplicates("loser_id").iterrows():
        if int(r["loser_id"]) not in names:
            names[int(r["loser_id"])] = str(r["loser_name"])

    pms = build_player_match_stats(matches[matches["match_date"] < reference_date])

    ratings: Dict[Tuple[int, str], PlayerRating] = {}
    for pid in player_ids:
        pid = int(pid)
        for surface in surfaces:
            stats = compute_player_stats_fast(
                pms, pid, surface, window=window, decay=decay,
            )
            if stats is None or stats.n_matches < min_matches:
                continue

            from wta_markov import serve_strength, return_strength, pressure_rating
            ratings[(pid, surface)] = PlayerRating(
                player_id=pid,
                player_name=names.get(pid, f"ID_{pid}"),
                surface=surface,
                stats=stats,
                last_match_date=reference_date,
                serve_strength=serve_strength(stats),
                return_strength=return_strength(stats),
                pressure_rating=pressure_rating(stats),
            )

    return ratings


def get_player_stats(
    ratings: Dict[Tuple[int, str], PlayerRating],
    player_id: int,
    surface: str,
    fallback_surfaces: List[str] = None,
    surface_blend: float = 0.70,
) -> Optional[PlayerServeStats]:
    exact = ratings.get((player_id, surface))
    if exact is not None:
        return exact.stats

    if fallback_surfaces is None:
        fallback_surfaces = ["Hard", "Clay", "Grass"]

    all_stats = [ratings[(player_id, s)].stats for s in fallback_surfaces if (player_id, s) in ratings]
    if not all_stats:
        return None

    n = len(all_stats)
    return PlayerServeStats(
        first_serve_in_pct=sum(s.first_serve_in_pct for s in all_stats) / n,
        first_serve_won_pct=sum(s.first_serve_won_pct for s in all_stats) / n,
        second_serve_won_pct=sum(s.second_serve_won_pct for s in all_stats) / n,
        return_pts_won_pct=sum(s.return_pts_won_pct for s in all_stats) / n,
        bp_saved_pct=sum(s.bp_saved_pct for s in all_stats) / n,
        bp_converted_pct=sum(s.bp_converted_pct for s in all_stats) / n,
        ace_rate=sum(s.ace_rate for s in all_stats) / n,
        n_matches=sum(s.n_matches for s in all_stats),
    )


def save_ratings(ratings: Dict[Tuple[int, str], PlayerRating], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(ratings, f)


def load_ratings(path: str) -> Dict[Tuple[int, str], PlayerRating]:
    with open(path, "rb") as f:
        return pickle.load(f)
