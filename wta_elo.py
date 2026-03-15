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

    # {surface: {player_id: rating}}
    ratings: Dict[str, Dict[int, float]] = field(default_factory=lambda: defaultdict(dict))
    # {surface: {player_id: match_count}}
    counts: Dict[str, Dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    # Track how far we've processed
    last_processed_date: Optional[pd.Timestamp] = None

    def _k(self, player_id: int, surface: str) -> float:
        n = self.counts.get(surface, {}).get(player_id, 0)
        return max(self.k_floor, self.k_initial * math.exp(-self.k_decay_rate * n))

    def _get_rating(self, player_id: int, surface: str) -> float:
        return self.ratings[surface].get(player_id, self.initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(self, winner_id: int, loser_id: int, surface: str) -> None:
        """Update ratings after a match result."""
        r_w = self._get_rating(winner_id, surface)
        r_l = self._get_rating(loser_id, surface)

        e_w = self.expected_score(r_w, r_l)
        e_l = 1.0 - e_w

        k_w = self._k(winner_id, surface)
        k_l = self._k(loser_id, surface)

        self.ratings[surface][winner_id] = r_w + k_w * (1.0 - e_w)
        self.ratings[surface][loser_id] = r_l + k_l * (0.0 - e_l)

        self.counts[surface][winner_id] = self.counts[surface].get(winner_id, 0) + 1
        self.counts[surface][loser_id] = self.counts[surface].get(loser_id, 0) + 1

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
        """Get rating on surface, falling back to cross-surface average."""
        if player_id in self.ratings.get(surface, {}):
            return self.ratings[surface][player_id]

        # Fallback: average across all surfaces where player has a rating
        all_r = [
            self.ratings[s][player_id]
            for s in self.ratings
            if player_id in self.ratings[s]
        ]
        if all_r:
            return sum(all_r) / len(all_r)
        return None

    def build_from_history(self, df: pd.DataFrame, up_to_date: pd.Timestamp) -> None:
        """Process all matches chronologically up to (exclusive) up_to_date."""
        subset = df[df["match_date"] < up_to_date].sort_values("match_date")
        for _, row in subset.iterrows():
            self.update(int(row["winner_id"]), int(row["loser_id"]), row["surface"])
        self.last_processed_date = up_to_date

    def update_from_history(self, df: pd.DataFrame, from_date: pd.Timestamp) -> None:
        """Incrementally update with matches from from_date onward."""
        subset = df[df["match_date"] >= from_date].sort_values("match_date")
        for _, row in subset.iterrows():
            self.update(int(row["winner_id"]), int(row["loser_id"]), row["surface"])
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
