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


def _inverse_lambda_from_p(
    p_over_target: float, line_bk: float, model: str, k: float
) -> Optional[float]:
    """Inverse-solve λ such that P(X > line | NB/Poisson) ≈ p_over_target.
    Uses bisection over [0.1, 30]. Returns None if input invalid.
    Used to derive market-implied λ from offered odds + line."""
    if p_over_target is None or p_over_target <= 0.0 or p_over_target >= 1.0:
        return None
    floor_t = int(np.floor(line_bk))

    def _prob(lam: float) -> float:
        lam = max(1e-6, float(lam))
        if model == "poisson":
            return float(1.0 - poisson.cdf(floor_t, lam))
        p = k / (k + lam)
        return float(1.0 - nbinom.cdf(floor_t, k, p))

    lo, hi = 0.1, 30.0
    p_lo, p_hi = _prob(lo), _prob(hi)
    if p_over_target <= p_lo:
        return lo
    if p_over_target >= p_hi:
        return hi
    for _ in range(50):
        mid = (lo + hi) / 2.0
        p_mid = _prob(mid)
        if p_mid < p_over_target:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 3)


def _market_implied_lambda(
    offered_odds_over: Optional[float],
    line_bk: float,
    model: str,
    k: float,
    juice_assumption: float = 0.05,
) -> Optional[float]:
    """Estimate market-implied λ from a single-side offered odds + line.
    Approximates juice removal symmetrically (default 5% margin → split as +2.5pp shrinkage on over)."""
    if offered_odds_over is None or offered_odds_over <= 1.0:
        return None
    p_market_raw = 1.0 / float(offered_odds_over)
    # Approximate de-juice: shrink p_over by half the assumed margin
    p_fair = p_market_raw / (1.0 + juice_assumption / 2.0)
    p_fair = max(0.001, min(0.999, p_fair))
    return _inverse_lambda_from_p(p_fair, line_bk, model, k)


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


def _load_odds_map(
    path: str,
) -> Dict[Tuple[str, str, str, str, float], Dict[str, Optional[float]]]:
    """Returns map: (league, home, away, side, line) -> {"over": odd, "under": odd}.
    Required columns: league, home_team, away_team, side, line, odds_over.
    Optional column: odds_under (added 2026-04-28 — enables UNDER side EV computation)."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    required = {"league", "home_team", "away_team", "side", "line", "odds_over"}
    if not required.issubset(df.columns):
        return {}
    has_under = "odds_under" in df.columns
    out: Dict[Tuple[str, str, str, str, float], Dict[str, Optional[float]]] = {}
    for _, r in df.iterrows():
        odd_over = _to_float(r.get("odds_over"))
        odd_under = _to_float(r.get("odds_under")) if has_under else None
        # Skip row if neither side has valid odds
        if (odd_over is None or odd_over <= 0) and (odd_under is None or odd_under <= 0):
            continue
        key = (
            str(r["league"]).strip().upper(),
            _norm_team(r["home_team"]),
            _norm_team(r["away_team"]),
            str(r["side"]).strip().lower(),
            float(r["line"]),
        )
        out[key] = {
            "over": odd_over if (odd_over is not None and odd_over > 0) else None,
            "under": odd_under if (odd_under is not None and odd_under > 0) else None,
        }
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
    if absences_map:
        print(
            f"⚠️  WARNING: --absences-csv provided ({len(absences_map)} entries) but depletion "
            f"NOT in training pipeline. Inference will apply depletion multiplier the model "
            f"was never trained on. Recommend retrain with depletion before relying on this."
        )
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

        # AUDIT FIX (2026-04-28): order CLIP→BLEND must match training pipeline
        # Previous (BUG): blend then clip → distributional drift vs training
        # Now: clip first, then blend (matches train_sot_per_team.py:284-288)
        lam_h = float(np.clip(lam_h, _SC["lambda_min"], _SC["lambda_max"]))
        lam_a = float(np.clip(lam_a, _SC["lambda_min"], _SC["lambda_max"]))
        lam_h = _SC["blend_empirical"] * lam_h + _SC["blend_league_mean"] * mu_h
        lam_a = _SC["blend_empirical"] * lam_a + _SC["blend_league_mean"] * mu_a

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

                odds_pair = odds_map.get((league, home, away, side, float(line_bk)), {})
                offered = odds_pair.get("over") if odds_pair else None
                offered_under = odds_pair.get("under") if odds_pair else None
                implied = (1.0 / offered) if offered is not None and offered > 0 else None
                edge = (p_final - implied) if implied is not None else None

                # Sprint 1a — market-implied lambda + divergence
                lam_bk_model = round(lam * scale_league, 3)
                lambda_implied_market = _market_implied_lambda(
                    offered, float(line_bk), args.model, k
                )
                lambda_divergence = (
                    round(lam_bk_model - lambda_implied_market, 3)
                    if lambda_implied_market is not None
                    else None
                )

                # EV calculation — over and under sides (added 2026-04-28: ev_under)
                p_under = 1.0 - p_final
                ev_over = (
                    round(p_final * float(offered) - 1.0, 4)
                    if offered is not None and offered > 0
                    else None
                )
                ev_under = (
                    round(p_under * float(offered_under) - 1.0, 4)
                    if offered_under is not None and offered_under > 0
                    else None
                )

                # Sprint 1b — recommended_status enum (now considers BOTH sides)
                # Logic: verified_edge if EITHER over or under hits +3% EV threshold
                over_qualifies = (
                    offered is not None
                    and args.min_odds <= offered <= args.max_odds
                    and p_final >= args.min_prob
                    and ev_over is not None
                    and ev_over >= 0.03
                )
                under_qualifies = (
                    offered_under is not None
                    and args.min_odds <= offered_under <= args.max_odds
                    and p_under >= args.min_prob
                    and ev_under is not None
                    and ev_under >= 0.03
                )

                if offered is not None or offered_under is not None:
                    odds_source = "market"
                    if over_qualifies or under_qualifies:
                        recommended_status = "verified_edge"
                        recommended = True
                    else:
                        recommended_status = "no_edge"
                        recommended = False
                else:
                    odds_source = "missing"
                    if (
                        p_final >= args.min_prob
                        and fair_odds_over is not None
                        and fair_odds_over <= args.max_fair_odds
                    ):
                        recommended_status = "watchlist"
                        recommended = True  # kept True for backward compat
                    else:
                        recommended_status = "skip"
                        recommended = False

                # Determine which side has the edge for output clarity
                edge_side: Optional[str] = None
                if over_qualifies and under_qualifies:
                    edge_side = "over" if (ev_over or 0) >= (ev_under or 0) else "under"
                elif over_qualifies:
                    edge_side = "over"
                elif under_qualifies:
                    edge_side = "under"

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
                    "lambda_bk": lam_bk_model,
                    "lambda_implied_market": lambda_implied_market,  # NEW
                    "lambda_divergence": lambda_divergence,  # NEW (model - market)
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
                    "offered_odds_under": offered_under,  # NEW (2026-04-28)
                    "implied_probability": implied,
                    "edge": edge,
                    "ev_over": ev_over,  # P_real * cotă_over - 1
                    "ev_under": ev_under,  # NEW (2026-04-28): (1-p) * cotă_under - 1
                    "edge_side": edge_side,  # NEW: "over"/"under"/None — which side has +3% EV
                    "odds_source": odds_source,
                    "recommended_status": recommended_status,
                    "recommended": recommended,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        print("No SOT rows evaluated.")
        return 0

    out = out.sort_values(["p_over", "edge"], ascending=[False, False], na_position="last").reset_index(drop=True)
    rec = out[out["recommended"]].copy()
    # Sprint 1b — split by status into separate files
    watchlist = out[out["recommended_status"] == "watchlist"].copy()
    verified_edge = out[out["recommended_status"] == "verified_edge"].copy()

    eval_path = Path(f"simulations/SOT/evaluations/{args.series}.1_SOT_Evaluations.csv")
    rec_path = Path(f"simulations/SOT/recommendations/{args.series}.2_SOT_Recommendations.csv")
    watchlist_path = Path(f"simulations/SOT/recommendations/{args.series}.3_SOT_Watchlist.csv")
    verified_path = Path(f"simulations/SOT/recommendations/{args.series}.4_SOT_Verified_Edge.csv")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    clean_output_dir(eval_path.parent)
    clean_output_dir(rec_path.parent)
    out.to_csv(eval_path, index=False)
    rec.to_csv(rec_path, index=False)
    watchlist.to_csv(watchlist_path, index=False)
    verified_edge.to_csv(verified_path, index=False)

    # Sprint 1a — divergence summary report (per league)
    div_summary = (
        out.dropna(subset=["lambda_divergence"])
        .groupby("league")["lambda_divergence"]
        .agg(["mean", "std", "count"])
        .round(3)
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    print(f"Saved SOT evaluations:      {eval_path} rows={len(out)}")
    print(f"Saved SOT recommendations:  {rec_path} rows={len(rec)}")
    print(f"Saved SOT watchlist:        {watchlist_path} rows={len(watchlist)} (model picks, no odds)")
    print(f"Saved SOT verified_edge:    {verified_path} rows={len(verified_edge)} (real EV ≥ +3%)")
    print(f"Global scaling: {_SCALE:.2f}x | Per-league overrides: {len(_SCALE_PER_LEAGUE)}")
    print(f"Elo: {'ON' if _ELO_CFG.get('enabled') else 'OFF'} | Absences loaded: {len(absences_map)}")
    print(f"Calibration: {'ON' if args.use_calibration else 'OFF (raw NB)'} | Lines: {_LINES}")

    # Print divergence summary if any market data available
    if not div_summary.empty:
        print("\n=== λ DIVERGENCE (model - market) per league ===")
        print(div_summary.to_string(index=False))
    else:
        print("\n[INFO] λ divergence: no market odds available in odds-csv (all rows lambda_implied_market=None)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())