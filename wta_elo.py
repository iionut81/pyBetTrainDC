from __future__ import annotations

"""
wta_elo.py
Surface-specific Elo ratings for WTA players.

Each surface (Hard, Clay, Grass) maintains its own rating pool.
K-factor decays with experience so new players converge fast,
established players move slowly.
"""

import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass
class SurfaceElo:
    """Surface-specific Elo rating system."""

    initial_rating: float = 1500.0
    k_initial: float = 40.0
    k_floor: float = 10.0
    k_decay_rate: float = 0.05

    # Inactivity decay: after N days without a match, rating decays toward initial
    inactivity_threshold_days: int = 60   # start decaying after 60 days
    inactivity_decay_rate: float = 0.002  # per day beyond threshold

    # {surface: {player_id: rating}}
    ratings: Dict[str, Dict[int, float]] = field(default_factory=lambda: defaultdict(dict))
    # {surface: {player_id: match_count}}
    counts: Dict[str, Dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    # {surface: {player_id: last_match_date}} for inactivity decay
    last_match: Dict[str, Dict[int, pd.Timestamp]] = field(default_factory=lambda: defaultdict(dict))
    # Track how far we've processed
    last_processed_date: Optional[pd.Timestamp] = None

    def _k(self, player_id: int, surface: str) -> float:
        n = self.counts.get(surface, {}).get(player_id, 0)
        return max(self.k_floor, self.k_initial * math.exp(-self.k_decay_rate * n))

    def _get_rating(self, player_id: int, surface: str) -> float:
        return self.ratings[surface].get(player_id, self.initial_rating)

    def _apply_inactivity_decay(
        self, player_id: int, surface: str, current_date: Optional[pd.Timestamp],
    ) -> None:
        """Decay rating toward initial_rating if player hasn't played recently."""
        if current_date is None:
            return
        last = self.last_match.get(surface, {}).get(player_id)
        if last is None:
            return
        days_inactive = (current_date - last).days
        if days_inactive <= self.inactivity_threshold_days:
            return
        excess = days_inactive - self.inactivity_threshold_days
        decay = 1.0 - self.inactivity_decay_rate * excess  # shrink toward mean
        decay = max(0.50, min(1.0, decay))  # never decay more than 50%
        r = self._get_rating(player_id, surface)
        self.ratings[surface][player_id] = (
            self.initial_rating + decay * (r - self.initial_rating)
        )

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def _mov_multiplier(
        w_sets: int, l_sets: int, total_games: int, best_of: int = 3,
    ) -> float:
        """Margin-of-victory multiplier (WElo, Angelini et al. 2022).

        Dominant wins (6-0 6-0) get a higher multiplier than tight wins (7-6 7-6).
        Returns a value in [0.75, 1.40] that scales the K-factor.
        """
        if w_sets <= 0 or total_games <= 0:
            return 1.0

        # Game difference: how many more games did the winner take?
        # In a best-of-3 WTA match, max total games = ~39 (6-7 7-6 7-6)
        # Straightforward: use the ratio of games won by winner
        w_games = (total_games + (w_sets - l_sets)) / 2.0  # approximate
        game_ratio = w_games / total_games  # 0.5 = dead even, 0.75+ = dominant

        # Straight-sets bonus
        straight_sets = 1.0 if l_sets == 0 else 0.0

        # Multiplier: 1.0 at average dominance, higher for blowouts, lower for tight
        # Average game_ratio for a winner ≈ 0.57; a 6-0 6-0 gives ~0.86
        mov = 0.75 + 0.65 * (game_ratio - 0.50) + 0.10 * straight_sets
        return max(0.75, min(1.40, mov))

    def update(
        self,
        winner_id: int,
        loser_id: int,
        surface: str,
        w_sets: int = 0,
        l_sets: int = 0,
        total_games: int = 0,
        match_date: Optional[pd.Timestamp] = None,
    ) -> None:
        """Update ratings after a match result.

        If score details are provided, applies margin-of-victory weighting (WElo).
        Applies inactivity decay before updating if match_date is provided.
        """
        # Apply inactivity decay before updating
        if match_date is not None:
            self._apply_inactivity_decay(winner_id, surface, match_date)
            self._apply_inactivity_decay(loser_id, surface, match_date)

        r_w = self._get_rating(winner_id, surface)
        r_l = self._get_rating(loser_id, surface)

        e_w = self.expected_score(r_w, r_l)
        e_l = 1.0 - e_w

        k_w = self._k(winner_id, surface)
        k_l = self._k(loser_id, surface)

        # Margin-of-victory scaling
        mov = self._mov_multiplier(w_sets, l_sets, total_games)

        self.ratings[surface][winner_id] = r_w + k_w * mov * (1.0 - e_w)
        self.ratings[surface][loser_id] = r_l + k_l * mov * (0.0 - e_l)

        self.counts[surface][winner_id] = self.counts[surface].get(winner_id, 0) + 1
        self.counts[surface][loser_id] = self.counts[surface].get(loser_id, 0) + 1

        if match_date is not None:
            self.last_match[surface][winner_id] = match_date
            self.last_match[surface][loser_id] = match_date

    def predict(self, player_a_id: int, player_b_id: int, surface: str) -> Optional[float]:
        """P(player A wins) from Elo ratings on the given surface.

        Returns None if neither player has a rating on any surface.
        Falls back to cross-surface average if missing on target surface.
        """
        r_a = self._resolve_rating(player_a_id, surface)
        r_b = self._resolve_rating(player_b_id, surface)

        if r_a is None and r_b is None:
            return None
        if r_a is None:
            r_a = self.initial_rating
        if r_b is None:
            r_b = self.initial_rating

        return self.expected_score(r_a, r_b)

    def _resolve_rating(self, player_id: int, surface: str) -> Optional[float]:
        """Get blended rating: 50% surface-specific + 50% overall.

        Falls back to overall-only or surface-only if one is missing.
        """
        r_surface = self.ratings.get(surface, {}).get(player_id)
        r_overall = self.ratings.get("__ALL__", {}).get(player_id)

        if r_surface is not None and r_overall is not None:
            w = self.SURFACE_BLEND
            return w * r_surface + (1.0 - w) * r_overall
        if r_surface is not None:
            return r_surface
        if r_overall is not None:
            return r_overall
        return None

    # Surface blend weight: 50% surface-specific + 50% overall
    SURFACE_BLEND: float = 0.50

    def _update_from_row(self, row: pd.Series) -> None:
        """Extract score info from a history row and call update() on both surface + overall."""
        w_sets = int(row.get("w_sets", 0) or 0)
        l_sets = int(row.get("l_sets", 0) or 0)
        total_games = int(row.get("total_games", 0) or 0)
        w_id = int(row["winner_id"])
        l_id = int(row["loser_id"])
        surface = row["surface"]
        md = pd.Timestamp(row["match_date"]) if "match_date" in row.index else None
        # Update surface-specific Elo
        self.update(w_id, l_id, surface,
                    w_sets=w_sets, l_sets=l_sets, total_games=total_games,
                    match_date=md)
        # Update overall Elo (surface="__ALL__")
        self.update(w_id, l_id, "__ALL__",
                    w_sets=w_sets, l_sets=l_sets, total_games=total_games,
                    match_date=md)

    def build_from_history(self, df: pd.DataFrame, up_to_date: pd.Timestamp) -> None:
        """Process all matches chronologically up to (exclusive) up_to_date."""
        subset = df[df["match_date"] < up_to_date].sort_values("match_date")
        for _, row in subset.iterrows():
            self._update_from_row(row)
        self.last_processed_date = up_to_date

    def update_from_history(self, df: pd.DataFrame, from_date: pd.Timestamp) -> None:
        """Incrementally update with matches from from_date onward."""
        subset = df[df["match_date"] >= from_date].sort_values("match_date")
        for _, row in subset.iterrows():
            self._update_from_row(row)
        if not subset.empty:
            self.last_processed_date = subset["match_date"].max()

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "SurfaceElo":
        with open(path, "rb") as f:
            return pickle.load(f)

    def player_rating_summary(self, player_id: int) -> Dict[str, float]:
        """Debug helper: return all surface ratings for a player."""
        return {
            s: self.ratings[s].get(player_id, self.initial_rating)
            for s in ["Hard", "Clay", "Grass"]
        }
