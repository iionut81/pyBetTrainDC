"""
run_sot_daily.py

Daily Shots on Target (SOT) evaluations + recommendations.
Produces per-team SOT Over probabilities at lines 2.5 / 3.5 / 4.5 / 5.5
(bookmaker scale). Applies scaling_factor from CFG["sot"] to translate
Flashscore lambda -> bookmaker-scale lambda, then NB CDF.

Usage:
    python run_sot_daily.py --api-key KEY [--insecure]
"""
from __future__ import annotations

import argparse
import datetime as dt
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from config import CFG
from data_loader import clean_output_dir, fetch_fixtures_from_api
from fhg_calibration import apply_platt_logit
from team_registry import resolve_team_or_warn

_SC = CFG["sot"]
_LINES = list(_SC["lines"])
_SCALE = float(_SC["scaling_factor"])
_SCALE_PER_LEAGUE: Dict[str, float] = {
    str(k).lower(): float(v) for k, v in (_SC.get("scaling_per_league") or {}).items()
}
_ELO_CFG = _SC.get("elo", {"enabled": False})
_DEPLETION = _SC.get("depletion", {})


def _scale_for(league: str) -> float:
    return _SCALE_PER_LEAGUE.get(str(league).lower(), _SCALE)


def _load_elo_ratings() -> Dict[str, Dict[str, Dict[str, float]]]:
    path = Path("team_ratings.pkl")
    if not path.exists() or not _ELO_CFG.get("enabled"):
        return {}
    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    except Exception:
        return {}
    if not isinstance(raw, dict) or "leagues" not in raw:
        return {}
    leagues = raw["leagues"]
    items = leagues.items() if isinstance(leagues, dict) else leagues
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for lg, payload in items:
        teams_payload = payload.get("teams", {}) if isinstance(payload, dict) else {}
        out[str(lg).lower()] = {
            str(team).lower(): {
                "attack": float(v.get("attack", 0.0)),
                "defence": float(v.get("defence", 0.0)),
            }
            for team, v in teams_payload.items()
        }
    return out


def _elo_multiplier(
    ratings: Dict[str, Dict[str, Dict[str, float]]],
    league: str,
    attacker: str,
    defender: str,
) -> float:
    if not _ELO_CFG.get("enabled") or not ratings:
        return 1.0
    lg = str(league).lower()
    league_ratings = ratings.get(lg)
    if not league_ratings:
        return 1.0
    atk = league_ratings.get(str(attacker).lower(), {}).get("attack", 0.0)
    dfc = league_ratings.get(str(defender).lower(), {}).get("defence", 0.0)
    elasticity = float(_ELO_CFG.get("sot_elasticity", 0.15))
    mult = 1.0 + elasticity * (atk + dfc)
    return float(np.clip(mult, _ELO_CFG.get("clip_min", 0.65), _ELO_CFG.get("clip_max", 1.50)))


def _load_absences(path: str) -> Dict[Tuple[str, str], int]:
    """Optional CSV with columns: league, team, absent_count.
    Returns map (league_lower, team_normalized) -> absent_count."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}
    required = {"league", "team", "absent_count"}
    if not required.issubset(df.columns):
        return {}
    out: Dict[Tuple[str, str], int] = {}
    for _, r in df.iterrows():
        try:
            count = int(r["absent_count"])
        except (TypeError, ValueError):
            continue
        key = (str(r["league"]).strip().lower(), _norm_team(r["team"]))
        out[key] = count
    return out


def _depletion_multiplier(absent_count: int) -> float:
    """Lambda reduction from squad depletion. Returns 1.0 when below threshold."""
    threshold = int(_DEPLETION.get("threshold", 3))
    if absent_count < threshold:
        return 1.0
    per_absence = float(_DEPLETION.get("penalty_per_absence", 0.05))
    max_penalty = float(_DEPLETION.get("max_penalty", 0.25))
    penalty = min(per_absence * absent_count, max_penalty)
    return max(1e-6, 1.0 - penalty)


def _norm_team(name: object) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prob_over_bk(line_bk: float, lam_our: float, model: str, k: float, league: str = "") -> float:
    scale = _scale_for(league)
    lam_bk = max(1e-6, float(lam_our) * scale)
    floor_t = int(np.floor(line_bk))
    if model == "poisson":
        return float(1.0 - poisson.cdf(floor_t, lam_bk))
    p = k / (k + lam_bk)
    return float(1.0 - nbinom.cdf(floor_t, k, p))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily SOT Over/Under per-team recommendations.")
    p.add_argument("--api-key", required=True)
    p.add_argument("--target-date", default=dt.date.today().isoformat())
    p.add_argument("--profiles-csv", default="simulations/SOT/data/sot_team_profiles.csv")
    p.add_argument("--league-params-csv", default="simulations/SOT/data/sot_league_params.csv")
    p.add_argument("--calibration-csv", default="simulations/SOT/data/sot_calibration.csv")
    p.add_argument("--odds-csv", default="", help="Optional CSV with columns: league,home_team,away_team,side,line,odds_over")
    p.add_argument("--absences-csv", default="", help="Optional CSV with columns: league,team,absent_count")
    p.add_argument("--model", choices=["poisson", "nb"], default="nb")
    p.add_argument("--use-calibration", action="store_true", default=True, help="Apply Platt calibration (default ON; corrects 8-12pp overconfidence)")
    p.add_argument("--no-calibration", dest="use_calibration", action="store_false", help="Disable Platt calibration (raw NB only)")
    p.add_argument("--min-prob", type=float, default=_SC["min_probability"])
    p.add_argument("--min-odds", type=float, default=_SC["min_odds"])
    p.add_argument("--max-odds", type=float, default=_SC["max_odds"])
    p.add_argument("--max-fair-odds", type=float, default=_SC["max_fair_odds"])
    p.add_argument("--series", default="1")
    p.add_argument("--insecure", action="store_true")
    return p.parse_args()


def _load_odds_map(path: str) -> Dict[Tuple[str, str, str, str, float], float]:
    """Returns map: (league, home, away, side, line) -> odds_over."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    required = {"league", "home_team", "away_team", "side", "line", "odds_over"}
    if not required.issubset(df.columns):
        return {}
    out: Dict[Tuple[str, str, str, str, float], float] = {}
    for _, r in df.iterrows():
        odd = _to_float(r.get("odds_over"))
        if odd is None or odd <= 0:
            continue
        key = (
            str(r["league"]).strip().upper(),
            _norm_team(r["home_team"]),
            _norm_team(r["away_team"]),
            str(r["side"]).strip().lower(),
            float(r["line"]),
        )
        out[key] = odd
    return out


def _get_calibration(
    cal_df: pd.DataFrame, league: str, side: str, line: float
) -> Tuple[float, float]:
    if cal_df.empty:
        return 0.0, 1.0
    sel = cal_df[
        (cal_df["league"] == league.upper())
        & (cal_df["side"] == side)
        & (np.isclose(cal_df["line"], line))
    ]
    if sel.empty:
        sel = cal_df[
            (cal_df["league"] == "__GLOBAL__")
            & (cal_df["side"] == side)
            & (np.isclose(cal_df["line"], line))
        ]
    if sel.empty:
        return 0.0, 1.0
    return float(sel.iloc[0]["a"]), float(sel.iloc[0]["b"])


def main() -> int:
    args = parse_args()
    profiles_path = Path(args.profiles_csv)
    league_params_path = Path(args.league_params_csv)
    if not profiles_path.exists() or not league_params_path.exists():
        raise RuntimeError("Missing trained SOT artifacts. Run train_sot_per_team.py first.")

    profiles = pd.read_csv(profiles_path)
    league_params = pd.read_csv(league_params_path)
    if profiles.empty or league_params.empty:
        raise RuntimeError("Empty SOT profiles or league params.")

    for c in ("league", "team"):
        profiles[c] = profiles[c].astype(str).str.strip().str.lower()
    league_params["league"] = league_params["league"].astype(str).str.strip().str.upper()
    profiles["team"] = profiles.apply(
        lambda row: resolve_team_or_warn(row["team"], row["league"].upper()),
        axis=1,
    )

    cal_path = Path(args.calibration_csv)
    cal_df = pd.read_csv(cal_path) if cal_path.exists() else pd.DataFrame(columns=["league", "side", "line", "a", "b"])

    odds_map = _load_odds_map(args.odds_csv)
    absences_map = _load_absences(args.absences_csv)
    elo_ratings = _load_elo_ratings()

    api_url = f"https://v3.football.api-sports.io/fixtures?date={args.target_date}"
    fixtures = fetch_fixtures_from_api(api_url=api_url, api_key=args.api_key, verify_ssl=not args.insecure)
    if not fixtures:
        print("No fixtures loaded.")
        return 0

    rows: list[dict] = []
    for fx in fixtures:
        league = str(fx.league).strip().upper()
        home_raw = _norm_team(fx.home_team)
        away_raw = _norm_team(fx.away_team)
        home = resolve_team_or_warn(home_raw, league)
        away = resolve_team_or_warn(away_raw, league)

        lp = league_params[league_params["league"] == league]
        if lp.empty:
            continue
        mu_h = float(lp.iloc[0].get("mu_home", _SC["default_mu_team"]))
        mu_a = float(lp.iloc[0].get("mu_away", _SC["default_mu_team"]))
        k_h = float(lp.iloc[0].get("k_home", _SC["default_k"]))
        k_a = float(lp.iloc[0].get("k_away", _SC["default_k"]))
        tempo_h = float(lp.iloc[0].get("tempo_home", 1.0))
        tempo_a = float(lp.iloc[0].get("tempo_away", 1.0))

        pp = profiles[profiles["league"] == league.lower()]
        hprof = pp[pp["team"] == home]
        aprof = pp[pp["team"] == away]
        if hprof.empty or aprof.empty:
            missing = []
            if hprof.empty:
                missing.append(f"home={home!r}")
            if aprof.empty:
                missing.append(f"away={away!r}")
            print(f"  [SKIP] {league} {home_raw} vs {away_raw} — no profile for {', '.join(missing)}")
            continue

        hrow = hprof.iloc[0]
        arow = aprof.iloc[0]

        lam_h = ((float(hrow["h_for"]) + float(arow["a_against"])) / 2.0) * tempo_h
        lam_a = ((float(arow["a_for"]) + float(hrow["h_against"])) / 2.0) * tempo_a

        # Fix #3: Elo/class strength adjustment.
        elo_mult_h = _elo_multiplier(elo_ratings, league, home, away)
        elo_mult_a = _elo_multiplier(elo_ratings, league, away, home)
        lam_h *= elo_mult_h
        lam_a *= elo_mult_a

        # Fix #5: Squad depletion penalty (from optional absences CSV).
        absent_h = absences_map.get((league.lower(), home), 0)
        absent_a = absences_map.get((league.lower(), away), 0)
        dep_mult_h = _depletion_multiplier(absent_h)
        dep_mult_a = _depletion_multiplier(absent_a)
        lam_h *= dep_mult_h
        lam_a *= dep_mult_a

        lam_h = _SC["blend_empirical"] * lam_h + _SC["blend_league_mean"] * mu_h
        lam_a = _SC["blend_empirical"] * lam_a + _SC["blend_league_mean"] * mu_a
        lam_h = float(np.clip(lam_h, _SC["lambda_min"], _SC["lambda_max"]))
        lam_a = float(np.clip(lam_a, _SC["lambda_min"], _SC["lambda_max"]))

        scale_league = _scale_for(league)

        for side, team_name, lam, k, elo_mult, dep_mult, absent in (
            ("home", home, lam_h, k_h, elo_mult_h, dep_mult_h, absent_h),
            ("away", away, lam_a, k_a, elo_mult_a, dep_mult_a, absent_a),
        ):
            for line_bk in _LINES:
                p_raw = _prob_over_bk(line_bk, lam, args.model, k, league)
                if args.use_calibration:
                    a, b = _get_calibration(cal_df, league, side, float(line_bk))
                    p_final = float(apply_platt_logit(np.array([p_raw], dtype=float), a, b)[0])
                else:
                    p_final = p_raw

                fair_odds_over = (1.0 / p_final) if p_final > 0 else None
                p_under = 1.0 - p_final
                fair_odds_under = (1.0 / p_under) if p_under > 0 else None

                offered = odds_map.get((league, home, away, side, float(line_bk)))
                implied = (1.0 / offered) if offered is not None and offered > 0 else None
                edge = (p_final - implied) if implied is not None else None

                if offered is not None:
                    recommended = bool(
                        args.min_odds <= offered <= args.max_odds
                        and p_final >= args.min_prob
                        and edge is not None
                        and edge > 0
                    )
                    odds_source = "market"
                else:
                    recommended = bool(
                        p_final >= args.min_prob
                        and fair_odds_over is not None
                        and fair_odds_over <= args.max_fair_odds
                    )
                    odds_source = "missing"

                rows.append({
                    "run_date": dt.date.today().isoformat(),
                    "match_date": fx.match_date,
                    "league": league,
                    "home_team": home,
                    "away_team": away,
                    "side": side,
                    "team": team_name,
                    "line": float(line_bk),
                    "model": args.model,
                    "lambda_our": round(lam, 3),
                    "lambda_bk": round(lam * scale_league, 3),
                    "scaling_factor": scale_league,
                    "elo_multiplier": round(elo_mult, 3),
                    "depletion_multiplier": round(dep_mult, 3),
                    "absent_count": int(absent),
                    "k_dispersion": k,
                    "p_over_raw": p_raw,
                    "p_over": p_final,
                    "fair_odds_over": fair_odds_over,
                    "fair_odds_under": fair_odds_under,
                    "offered_odds_over": offered,
                    "implied_probability": implied,
                    "edge": edge,
                    "odds_source": odds_source,
                    "recommended": recommended,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        print("No SOT rows evaluated.")
        return 0

    out = out.sort_values(["p_over", "edge"], ascending=[False, False], na_position="last").reset_index(drop=True)
    rec = out[out["recommended"]].copy()

    eval_path = Path(f"simulations/SOT/evaluations/{args.series}.1_SOT_Evaluations.csv")
    rec_path = Path(f"simulations/SOT/recommendations/{args.series}.2_SOT_Recommendations.csv")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    clean_output_dir(eval_path.parent)
    clean_output_dir(rec_path.parent)
    out.to_csv(eval_path, index=False)
    rec.to_csv(rec_path, index=False)

    print(f"Saved SOT evaluations:      {eval_path} rows={len(out)}")
    print(f"Saved SOT recommendations:  {rec_path} rows={len(rec)}")
    print(f"Global scaling: {_SCALE:.2f}x | Per-league overrides: {len(_SCALE_PER_LEAGUE)}")
    print(f"Elo: {'ON' if _ELO_CFG.get('enabled') else 'OFF'} | Absences loaded: {len(absences_map)}")
    print(f"Calibration: {'ON' if args.use_calibration else 'OFF (raw NB)'} | Lines: {_LINES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())