"""Canonical team name registry.

Loads team_ids.yaml and provides deterministic name resolution.
Falls back to fuzzy matching only for genuinely unknown teams,
logging a warning so the alias can be added.

Usage:
    from team_registry import resolve_team
    canonical = resolve_team("manchester city", "E0")  # -> "man city"
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import yaml

_YAML_PATH = Path(__file__).resolve().parent / "team_ids.yaml"

# (league_upper, alias_lower) -> canonical_name
_ALIAS_MAP: Dict[Tuple[str, str], str] = {}

# league_upper -> set of canonical names
_CANONICAL_NAMES: Dict[str, Set[str]] = {}

_loaded = False


def _norm(name: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(str(name).strip().lower().split())


def _load() -> None:
    global _loaded
    if _loaded:
        return
    with open(_YAML_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    for league_key, aliases in data.items():
        if not isinstance(aliases, dict):
            continue
        league = league_key.strip().upper()
        if league.startswith("_"):
            # _global or other special sections
            for alias, canonical in aliases.items():
                _ALIAS_MAP[("*", _norm(alias))] = _norm(canonical)
            continue
        if league not in _CANONICAL_NAMES:
            _CANONICAL_NAMES[league] = set()
        for alias, canonical in aliases.items():
            canon = _norm(str(canonical))
            _ALIAS_MAP[(league, _norm(str(alias)))] = canon
            _CANONICAL_NAMES[league].add(canon)

    _loaded = True


def reload() -> None:
    """Force reload of team_ids.yaml (useful after edits)."""
    global _loaded
    _ALIAS_MAP.clear()
    _CANONICAL_NAMES.clear()
    _loaded = False
    _load()


def resolve_team(name: str, league: str) -> Optional[str]:
    """Resolve a team name to its canonical form.

    Returns the canonical name if found in the alias table,
    or the normalized name itself if it's already canonical.
    Returns None if the team is completely unknown.
    """
    _load()
    league_upper = league.strip().upper()
    normed = _norm(name)

    # 1. Direct alias lookup
    canonical = _ALIAS_MAP.get((league_upper, normed))
    if canonical is not None:
        return canonical

    # 2. Global alias lookup
    canonical = _ALIAS_MAP.get(("*", normed))
    if canonical is not None:
        return canonical

    # 3. Already a canonical name
    if normed in _CANONICAL_NAMES.get(league_upper, set()):
        return normed

    # 4. Unknown — return None (caller decides whether to fuzzy-match)
    return None


def resolve_team_or_warn(name: str, league: str) -> str:
    """Resolve with a warning if falling back to raw name.

    Use this in daily pipelines so unrecognized names are logged
    and can be added to team_ids.yaml.
    """
    result = resolve_team(name, league)
    if result is not None:
        return result
    normed = _norm(name)
    print(f"  [WARN] Unknown team: {name!r} in {league} — add to team_ids.yaml")
    return normed


def get_canonical_names(league: str) -> Set[str]:
    """Return all canonical names for a league."""
    _load()
    return set(_CANONICAL_NAMES.get(league.strip().upper(), set()))


# --- CLI: check mode ---
if __name__ == "__main__":
    _load()
    if "--check" in sys.argv:
        print(f"Loaded {len(_ALIAS_MAP)} aliases across {len(_CANONICAL_NAMES)} leagues.\n")
        for league in sorted(_CANONICAL_NAMES):
            n_aliases = sum(1 for (lg, _) in _ALIAS_MAP if lg == league)
            n_canonical = len(_CANONICAL_NAMES[league])
            print(f"  {league}: {n_canonical} canonical, {n_aliases} aliases")
    elif "--test" in sys.argv:
        # Quick smoke test
        tests = [
            ("manchester city", "E0", "man city"),
            ("man city", "E0", "man city"),
            ("hull", "E1", "hull city"),
            ("sheffield weds", "E1", "sheff wed"),
            ("fc bayern münchen", "D1", "bayern munich"),
            ("paris sg", "F1", "psg"),
            ("sp lisbon", "P1", "sporting"),
        ]
        ok = 0
        for name, league, expected in tests:
            result = resolve_team(name, league)
            status = "OK" if result == expected else f"FAIL (got {result!r})"
            print(f"  {status}: resolve({name!r}, {league!r}) -> {result!r}")
            if result == expected:
                ok += 1
        print(f"\n{ok}/{len(tests)} passed")