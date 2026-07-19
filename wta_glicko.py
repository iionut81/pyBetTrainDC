from __future__ import annotations

"""
wta_glicko.py
Surface-specific Glicko-2 ratings for WTA players (production version).

Validated via backtest_glicko_vs_elo_blended.py against production SurfaceElo:
~10.8% lower Brier score, ~29% lower log-loss, on 41,631 historical matches
(consistent improvement across hot-streak and non-hot-streak matches alike).

Same 50% surface + 50% overall blend as SurfaceElo._resolve_rating, so the
two systems are directly comparable / swappable in the pipeline.
"""

import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import glicko2
import pandas as pd

Q = math.log(10) / 400.0


@dataclass
class SurfaceGlicko:
    """Surface-specific Glicko-2 rating system, blended with an overall pool."""

    surface_blend: float = 0.50  # matches SurfaceElo.SURFACE_BLEND

    # {surface: {player_id: glicko2.Player}} — "__ALL__" is the cross-surface pool
    ratings: Dict[str, Dict[int, "glicko2.Player"]] = field(default_factory=lambda: defaultdict(dict))
    last_processed_date: Optional[pd.Timestamp] = None

    def _get(self, player_id: int, surface: str) -> "glicko2.Player":
        pool = self.ratings[surface]
        if player_id not in pool:
            pool[player_id] = glicko2.Player()
        return pool[player_id]

    @staticmethod
    def _g(rd: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * Q**2 * rd**2 / math.pi**2)

    def _blended(self, player_id: int, surface: str) -> tuple[float, float]:
        s = self._get(player_id, surface)
        a = self._get(player_id, "__ALL__")
        r = self.surface_blend * s.rating + (1.0 - self.surface_blend) * a.rating
        rd = self.surface_blend * s.rd + (1.0 - self.surface_blend) * a.rd
        return r, rd

    def predict(self, player_a_id: int, player_b_id: int, surface: str) -> Optional[float]:
        """P(player A wins), blended surface+overall rating (same shape as SurfaceElo.predict)."""
        r_a, _ = self._blended(player_a_id, surface)
        r_b, rd_b = self._blended(player_b_id, surface)
        g = self._g(rd_b)
        return 1.0 / (1.0 + 10.0 ** (-g * (r_a - r_b) / 400.0))

    def update(
        self,
        winner_id: int,
        loser_id: int,
        surface: str,
        match_date: Optional[pd.Timestamp] = None,
    ) -> None:
        """Update surface-specific AND overall Glicko-2 ratings for both players.

        One match = one rating period (standard simplification used in the
        validated backtest). No margin-of-victory weighting (not tested).
        """
        sw, sl = self._get(winner_id, surface), self._get(loser_id, surface)
        aw, al = self._get(winner_id, "__ALL__"), self._get(loser_id, "__ALL__")

        sw.update_player([sl.rating], [sl.rd], [1])
        sl.update_player([sw.rating], [sw.rd], [0])
        aw.update_player([al.rating], [al.rd], [1])
        al.update_player([aw.rating], [aw.rd], [0])

        if match_date is not None:
            self.last_processed_date = match_date

    def _update_from_row(self, row: pd.Series) -> None:
        w_id, l_id = int(row["winner_id"]), int(row["loser_id"])
        surface = row["surface"]
        md = pd.Timestamp(row["match_date"]) if "match_date" in row.index else None
        self.update(w_id, l_id, surface, match_date=md)

    def build_from_history(self, df: pd.DataFrame, up_to_date: pd.Timestamp) -> None:
        subset = df[df["match_date"] < up_to_date].sort_values("match_date")
        for _, row in subset.iterrows():
            self._update_from_row(row)
        self.last_processed_date = up_to_date

    def update_from_history(self, df: pd.DataFrame, from_date: pd.Timestamp) -> None:
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
    def load(path: str) -> "SurfaceGlicko":
        with open(path, "rb") as f:
            return pickle.load(f)

    def player_rating_summary(self, player_id: int) -> Dict[str, float]:
        """Debug helper: blended rating per surface for a player."""
        return {s: self._blended(player_id, s)[0] for s in ["Hard", "Clay", "Grass"]}
