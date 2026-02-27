from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from scipy.optimize import minimize
from scipy.stats import poisson


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
MAX_STATSCHECKER_RESULTS_PAGES = 25
MAX_STATSCHECKER_FIXTURE_PAGES = 49
DEFAULT_MIN_TEAMS_PER_LEAGUE = 4
DEFAULT_MIN_MATCHES_PER_LEAGUE = 12
HISTORY_STORE_DEFAULT = "data/historical/historical_matches.csv"
HTTP_CACHE_DIR = ".http_cache"
HTTP_CACHE_TTL_SECONDS = 6 * 3600
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_HTTP_RETRIES = 4
BACKOFF_BASE_SECONDS = 2.0
MIN_REQUEST_INTERVAL_BY_SOURCE = {
    "footystats": 1.0,
    "scores24": 1.0,
    "statschecker": 2.5,
}
_LAST_REQUEST_TS_BY_SOURCE: Dict[str, float] = {}
SIMULATIONS_DIR = "simulations/backtests"
SIMULATION_LOG_FILE = "simulation_log.csv"


@dataclasses.dataclass
class MatchRecord:
    source: str
    league: str
    match_date: dt.date
    home_team: str
    away_team: str
    home_goals: Optional[int]
    away_goals: Optional[int]
    odds_1x: Optional[float]
    odds_x2: Optional[float]

    @property
    def is_finished(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None


@dataclasses.dataclass
class Prediction:
    match_date: dt.date
    league: str
    home_team: str
    away_team: str
    market: str  # 1X or X2
    model_probability: float
    fair_odds: float
    offered_odds: Optional[float]
    edge: Optional[float]
    result: Optional[bool]


def normalize_team_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    return cleaned


def safe_float(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    val = raw.strip().replace(",", ".")
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def safe_int(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None


def _cache_paths(url: str) -> Tuple[str, str]:
    os.makedirs(HTTP_CACHE_DIR, exist_ok=True)
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return (
        os.path.join(HTTP_CACHE_DIR, f"{key}.html"),
        os.path.join(HTTP_CACHE_DIR, f"{key}.meta.json"),
    )


def _read_cached_html(url: str, max_age_seconds: int, allow_stale: bool = False) -> Optional[str]:
    html_path, meta_path = _cache_paths(url)
    if not (os.path.exists(html_path) and os.path.exists(meta_path)):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        fetched_at = float(meta.get("fetched_at", 0.0))
    except Exception:
        return None
    age = time.time() - fetched_at
    if age <= max_age_seconds or allow_stale:
        try:
            with open(html_path, "r", encoding="utf-8") as fh:
                return fh.read()
        except Exception:
            return None
    return None


def _write_cached_html(url: str, text: str, status_code: int) -> None:
    html_path, meta_path = _cache_paths(url)
    try:
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"url": url, "status_code": status_code, "fetched_at": time.time()}, fh)
    except Exception:
        return


def fetch_html(
    session: requests.Session,
    url: str,
    source: str,
    timeout: int = 12,
    cache_ttl_seconds: int = HTTP_CACHE_TTL_SECONDS,
) -> str:
    cached = _read_cached_html(url, max_age_seconds=cache_ttl_seconds)
    if cached is not None:
        return cached

    min_interval = MIN_REQUEST_INTERVAL_BY_SOURCE.get(source, 1.0)
    last_ts = _LAST_REQUEST_TS_BY_SOURCE.get(source, 0.0)
    wait = min_interval - (time.time() - last_ts)
    if wait > 0:
        time.sleep(wait)

    last_error: Optional[str] = None
    for attempt in range(MAX_HTTP_RETRIES):
        try:
            resp = session.get(url, timeout=timeout)
            _LAST_REQUEST_TS_BY_SOURCE[source] = time.time()
            if resp.status_code == 200:
                _write_cached_html(url, resp.text, resp.status_code)
                return resp.text
            if resp.status_code in RETRYABLE_STATUS_CODES:
                last_error = f"HTTP {resp.status_code}"
                sleep_for = BACKOFF_BASE_SECONDS * (2**attempt) + random.uniform(0.0, 0.4)
                print(
                    f"[WARN][{source}] {url} -> {resp.status_code}, retry {attempt + 1}/{MAX_HTTP_RETRIES}"
                )
                time.sleep(sleep_for)
                continue
            raise RuntimeError(f"{source} request failed with HTTP {resp.status_code} for {url}")
        except requests.RequestException as exc:
            last_error = str(exc)
            sleep_for = BACKOFF_BASE_SECONDS * (2**attempt) + random.uniform(0.0, 0.4)
            print(f"[WARN][{source}] {url} network error, retry {attempt + 1}/{MAX_HTTP_RETRIES}: {exc}")
            time.sleep(sleep_for)

    stale = _read_cached_html(url, max_age_seconds=cache_ttl_seconds, allow_stale=True)
    if stale is not None:
        print(f"[WARN][{source}] Using stale cache for {url} after failures ({last_error}).")
        return stale
    raise RuntimeError(f"{source} failed to fetch {url} after retries ({last_error}).")
    txt = raw.strip()
    if txt == "":
        return None
    try:
        return int(txt)
    except ValueError:
        return None


class BaseScraper:
    source_name: str

    def fetch(self, start_date: dt.date, end_date: dt.date) -> List[MatchRecord]:
        raise NotImplementedError


class FootyStatsScraper(BaseScraper):
    source_name = "footystats"

    def __init__(self, session: requests.Session):
        self.session = session

    def fetch(self, start_date: dt.date, end_date: dt.date) -> List[MatchRecord]:
        # FootyStats uses dynamic pages. We attempt extracting structured JSON from listing pages.
        # If selectors change, update parse_* methods below.
        url = "https://footystats.org/"
        html = fetch_html(self.session, url, self.source_name, timeout=20)
        soup = BeautifulSoup(html, "html.parser")
        scripts = soup.find_all("script")
        out: List[MatchRecord] = []

        for script in scripts:
            if not script.string:
                continue
            if "application/ld+json" in str(script.get("type", "")):
                try:
                    payload = json.loads(script.string)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    recs = self._records_from_ldjson(payload, start_date, end_date)
                    out.extend(recs)
                elif isinstance(payload, list):
                    for node in payload:
                        if isinstance(node, dict):
                            out.extend(self._records_from_ldjson(node, start_date, end_date))
        return out

    def _records_from_ldjson(
        self, node: dict, start_date: dt.date, end_date: dt.date
    ) -> List[MatchRecord]:
        records: List[MatchRecord] = []
        if node.get("@type") not in {"SportsEvent", "Event"}:
            return records
        start = node.get("startDate")
        if not start:
            return records
        try:
            event_date = dateparser.parse(start).date()
        except Exception:
            return records
        if not (start_date <= event_date <= end_date):
            return records

        home = node.get("homeTeam", {})
        away = node.get("awayTeam", {})
        home_name = (home.get("name") or "").strip()
        away_name = (away.get("name") or "").strip()
        if not home_name or not away_name:
            return records

        home_goals = None
        away_goals = None
        odds_1x = None
        odds_x2 = None

        result = node.get("result")
        if isinstance(result, dict):
            home_goals = safe_int(str(result.get("homeTeamScore", "") or ""))
            away_goals = safe_int(str(result.get("awayTeamScore", "") or ""))

        offers = node.get("offers")
        if isinstance(offers, dict):
            # Most pages do not expose double-chance prices in schema; keep placeholder.
            pass

        records.append(
            MatchRecord(
                source=self.source_name,
                league=(node.get("name") or "unknown").strip(),
                match_date=event_date,
                home_team=home_name,
                away_team=away_name,
                home_goals=home_goals,
                away_goals=away_goals,
                odds_1x=odds_1x,
                odds_x2=odds_x2,
            )
        )
        return records


class Scores24Scraper(BaseScraper):
    source_name = "scores24"

    def __init__(self, session: requests.Session):
        self.session = session

    def fetch(self, start_date: dt.date, end_date: dt.date) -> List[MatchRecord]:
        # Public HTML fallback. If blocked by anti-bot, run with prepared local dumps.
        # Cap crawl window to avoid one-request-per-day timeouts on large backtests.
        max_days = 45
        if (end_date - start_date).days + 1 > max_days:
            start_date = end_date - dt.timedelta(days=max_days - 1)
        out: List[MatchRecord] = []
        cur = start_date
        while cur <= end_date:
            url = f"https://scores24.live/en/soccer?date={cur.isoformat()}"
            try:
                html = fetch_html(self.session, url, self.source_name, timeout=10)
            except Exception:
                cur += dt.timedelta(days=1)
                continue
            out.extend(self._parse_day_page(html, cur))
            cur += dt.timedelta(days=1)
        return out

    def _parse_day_page(self, html: str, day: dt.date) -> List[MatchRecord]:
        soup = BeautifulSoup(html, "html.parser")
        records: List[MatchRecord] = []
        cards = soup.select("[data-event-id]")
        for card in cards:
            home = card.select_one("[data-home-team-name]")
            away = card.select_one("[data-away-team-name]")
            score = card.select_one("[data-score]")
            if home is None or away is None:
                continue
            home_name = home.get_text(strip=True)
            away_name = away.get_text(strip=True)
            hg = ag = None
            if score is not None:
                parts = re.findall(r"\d+", score.get_text(" ", strip=True))
                if len(parts) >= 2:
                    hg, ag = safe_int(parts[0]), safe_int(parts[1])

            odds_1x = None
            odds_x2 = None
            odd_1x_node = card.select_one("[data-odd-1x]")
            odd_x2_node = card.select_one("[data-odd-x2]")
            if odd_1x_node:
                odds_1x = safe_float(odd_1x_node.get_text(strip=True))
            if odd_x2_node:
                odds_x2 = safe_float(odd_x2_node.get_text(strip=True))

            league_node = card.select_one("[data-league-name]")
            league = league_node.get_text(strip=True) if league_node else "unknown"

            records.append(
                MatchRecord(
                    source=self.source_name,
                    league=league,
                    match_date=day,
                    home_team=home_name,
                    away_team=away_name,
                    home_goals=hg,
                    away_goals=ag,
                    odds_1x=odds_1x,
                    odds_x2=odds_x2,
                )
            )
        return records


class StatsCheckerScraper(BaseScraper):
    source_name = "statschecker"

    def __init__(self, session: requests.Session):
        self.session = session

    def fetch(self, start_date: dt.date, end_date: dt.date) -> List[MatchRecord]:
        out: List[MatchRecord] = []
        result_pages = self._result_pages()[:MAX_STATSCHECKER_RESULTS_PAGES]

        for url in result_pages:
            try:
                out.extend(self._parse_results_page(url, start_date, end_date))
            except Exception:
                continue
        return out

    def fetch_fixtures_for_date(self, target_date: dt.date) -> List[MatchRecord]:
        fixtures: List[MatchRecord] = []
        fixture_pages = self._fixture_pages()[:MAX_STATSCHECKER_FIXTURE_PAGES]
        parsed_pages = 0
        failed_pages = 0
        consecutive_failures = 0
        max_consecutive_failures = 6
        for url in fixture_pages:
            try:
                fixtures.extend(self._parse_fixtures_page(url, target_date))
                parsed_pages += 1
                consecutive_failures = 0
            except Exception:
                failed_pages += 1
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(
                        f"[WARN] statschecker fixtures circuit-breaker triggered after "
                        f"{consecutive_failures} consecutive failures."
                    )
                    break
                continue
        print(
            f"[INFO] statschecker fixtures for {target_date.isoformat()}: "
            f"pages_ok={parsed_pages}, pages_failed={failed_pages}, scanned={parsed_pages + failed_pages}, "
            f"matches={len(fixtures)}"
        )
        return fixtures

    def _homepage_soup(self) -> Optional[BeautifulSoup]:
        homepage = "https://www.statschecker.com/"
        try:
            html = fetch_html(self.session, homepage, self.source_name, timeout=15)
        except Exception:
            return None
        return BeautifulSoup(html, "html.parser")

    def _result_pages(self) -> List[str]:
        soup = self._homepage_soup()
        if soup is None:
            return []
        links: List[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if "/results/" not in href:
                continue
            if not href.startswith("http"):
                href = "https://www.statschecker.com" + href
            links.append(href)
        return sorted(set(links))

    def _fixture_pages(self) -> List[str]:
        soup = self._homepage_soup()
        if soup is None:
            return []
        links: List[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if "/fixtures/" not in href:
                continue
            if "#" in href:
                continue
            if not href.startswith("http"):
                href = "https://www.statschecker.com" + href
            links.append(href)
        return sorted(set(links))

    def _parse_results_page(
        self, url: str, start_date: dt.date, end_date: dt.date
    ) -> List[MatchRecord]:
        html = fetch_html(self.session, url, self.source_name, timeout=15)
        soup = BeautifulSoup(html, "html.parser")
        h1 = soup.find("h1")
        league_name = h1.get_text(" ", strip=True) if h1 else "unknown"
        league_name = re.sub(r"\s+Results$", "", league_name, flags=re.IGNORECASE)

        records: List[MatchRecord] = []
        for row in soup.select("tr"):
            tds = row.find_all("td")
            if len(tds) != 4:
                continue

            team_links = tds[0].find_all("a")
            if len(team_links) < 2:
                continue
            home_team = team_links[0].get_text(" ", strip=True)
            away_team = team_links[1].get_text(" ", strip=True)
            if not home_team or not away_team:
                continue

            nums = re.findall(r"\d+", tds[1].get_text(" ", strip=True))
            if len(nums) < 2:
                continue
            home_goals = safe_int(nums[0])
            away_goals = safe_int(nums[1])

            timestamp = tds[2].get("data-sort") or tds[3].get("data-sort")
            match_date = None
            if timestamp and str(timestamp).isdigit():
                match_date = dt.datetime.fromtimestamp(int(timestamp), dt.UTC).date()
            else:
                date_text = tds[2].get_text(" ", strip=True)
                time_text = tds[3].get_text(" ", strip=True)
                try:
                    match_date = dateparser.parse(f"{date_text} {time_text}", fuzzy=True).date()
                except Exception:
                    continue

            if not (start_date <= match_date <= end_date):
                continue

            records.append(
                MatchRecord(
                    source=self.source_name,
                    league=league_name,
                    match_date=match_date,
                    home_team=home_team,
                    away_team=away_team,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    odds_1x=None,
                    odds_x2=None,
                )
            )
        return records

    def _parse_fixtures_page(self, url: str, target_date: dt.date) -> List[MatchRecord]:
        html = fetch_html(self.session, url, self.source_name, timeout=15)
        soup = BeautifulSoup(html, "html.parser")
        h1 = soup.find("h1")
        league_name = h1.get_text(" ", strip=True) if h1 else "unknown"
        league_name = re.sub(r"\s+Fixtures$", "", league_name, flags=re.IGNORECASE)

        out: List[MatchRecord] = []
        for row in soup.select("tr"):
            tds = row.find_all("td")
            if len(tds) < 3:
                continue
            team_links = tds[0].find_all("a")
            if len(team_links) < 2:
                continue
            timestamp = tds[1].get("data-sort")
            if not timestamp or not str(timestamp).isdigit():
                continue
            match_date = dt.datetime.fromtimestamp(int(timestamp), dt.UTC).date()
            if match_date != target_date:
                continue
            home_team = team_links[0].get_text(" ", strip=True)
            away_team = team_links[1].get_text(" ", strip=True)
            if not home_team or not away_team:
                continue
            out.append(
                MatchRecord(
                    source=self.source_name,
                    league=league_name,
                    match_date=match_date,
                    home_team=home_team,
                    away_team=away_team,
                    home_goals=None,
                    away_goals=None,
                    odds_1x=None,
                    odds_x2=None,
                )
            )
        return out


class DixonColesModel:
    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        self.teams: List[str] = []
        self.attack: np.ndarray = np.array([])
        self.defense: np.ndarray = np.array([])
        self.home_adv: float = 0.0
        self.rho: float = 0.0
        self._team_to_idx: Dict[str, int] = {}

    @staticmethod
    def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
        if x == 0 and y == 0:
            return 1 - (lam * mu * rho)
        if x == 0 and y == 1:
            return 1 + (lam * rho)
        if x == 1 and y == 0:
            return 1 + (mu * rho)
        if x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def fit(self, matches: pd.DataFrame, decay_xi: float = 0.0, reference_date: Optional[dt.date] = None) -> None:
        teams = sorted(set(matches["home_team"]).union(set(matches["away_team"])))
        if len(teams) < 2:
            raise ValueError("Need at least two teams to fit Dixon-Coles model.")
        self.teams = teams
        self._team_to_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        home_idx = matches["home_team"].map(self._team_to_idx).to_numpy(dtype=int)
        away_idx = matches["away_team"].map(self._team_to_idx).to_numpy(dtype=int)
        hg = matches["home_goals"].to_numpy(dtype=int)
        ag = matches["away_goals"].to_numpy(dtype=int)
        weights = np.ones(len(matches), dtype=float)
        if decay_xi > 0.0 and "match_date" in matches.columns:
            md = pd.to_datetime(matches["match_date"], errors="coerce").dt.date
            ref = reference_date or max((d for d in md if d is not None), default=None)
            if ref is not None:
                age_days = np.array(
                    [(ref - d).days if d is not None else 0 for d in md],
                    dtype=float,
                )
                age_days = np.clip(age_days, 0.0, None)
                weights = np.exp(-decay_xi * age_days)
                weights = np.clip(weights, 1e-8, None)

        def nll(params: np.ndarray) -> float:
            attack = params[:n]
            defense = params[n : 2 * n]
            home_adv = params[-2]
            rho = params[-1]

            # Identification constraint.
            attack = attack - np.mean(attack)

            lam = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
            mu = np.exp(attack[away_idx] + defense[home_idx])

            tau = np.array([self._tau(x, y, l, m, rho) for x, y, l, m in zip(hg, ag, lam, mu)])
            tau = np.clip(tau, 1e-8, None)

            log_p = np.log(tau)
            log_p += poisson.logpmf(hg, lam)
            log_p += poisson.logpmf(ag, mu)
            return -np.sum(weights * log_p)

        x0 = np.zeros(2 * n + 2)
        x0[-2] = 0.1
        x0[-1] = -0.05

        bounds = [(-3.0, 3.0)] * (2 * n) + [(-1.0, 1.0), (-0.2, 0.2)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 120})
        if not res.success:
            raise RuntimeError(f"Dixon-Coles fit failed: {res.message}")

        params = res.x
        attack = params[:n] - np.mean(params[:n])
        defense = params[n : 2 * n]

        self.attack = attack
        self.defense = defense
        self.home_adv = params[-2]
        self.rho = params[-1]

    def predict_1x_x2(self, home_team: str, away_team: str) -> Tuple[float, float]:
        if home_team not in self._team_to_idx or away_team not in self._team_to_idx:
            raise KeyError(f"Unknown team in fitted model: {home_team} vs {away_team}")
        hi = self._team_to_idx[home_team]
        ai = self._team_to_idx[away_team]
        lam = math.exp(self.attack[hi] + self.defense[ai] + self.home_adv)
        mu = math.exp(self.attack[ai] + self.defense[hi])
        mat = self._score_matrix(lam, mu, self.rho, self.max_goals)
        p_home = np.tril(mat, k=-1).sum()
        p_draw = np.trace(mat)
        p_away = np.triu(mat, k=1).sum()
        p_1x = float(p_home + p_draw)
        p_x2 = float(p_draw + p_away)
        return p_1x, p_x2

    def _score_matrix(self, lam: float, mu: float, rho: float, max_goals: int) -> np.ndarray:
        h = poisson.pmf(np.arange(max_goals + 1), lam)
        a = poisson.pmf(np.arange(max_goals + 1), mu)
        base = np.outer(h, a)
        for x in (0, 1):
            for y in (0, 1):
                base[x, y] *= self._tau(x, y, lam, mu, rho)
        total = base.sum()
        if total <= 0:
            return base
        return base / total


def deduplicate_records(records: Sequence[MatchRecord]) -> List[MatchRecord]:
    seen = {}
    for r in records:
        key = (
            r.match_date,
            normalize_team_name(r.home_team),
            normalize_team_name(r.away_team),
        )
        # Prefer rows with odds and final score.
        score = 0
        if r.odds_1x is not None or r.odds_x2 is not None:
            score += 1
        if r.is_finished:
            score += 2
        prev = seen.get(key)
        if prev is None:
            seen[key] = (score, r)
        elif score >= prev[0]:
            seen[key] = (score, r)
    return [v[1] for v in seen.values()]


def records_to_df(records: Sequence[MatchRecord]) -> pd.DataFrame:
    base_cols = [
        "source",
        "league",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "odds_1x",
        "odds_x2",
    ]
    rows = []
    for r in records:
        rows.append(
            {
                "source": r.source,
                "league": r.league,
                "match_date": r.match_date,
                "home_team": normalize_team_name(r.home_team),
                "away_team": normalize_team_name(r.away_team),
                "home_goals": r.home_goals,
                "away_goals": r.away_goals,
                "odds_1x": r.odds_1x,
                "odds_x2": r.odds_x2,
            }
        )
    df = pd.DataFrame(rows, columns=base_cols)
    if df.empty:
        return df
    df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    return df


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    # Corporate/network SSL interception can break cert validation in default Python trust stores.
    s.verify = False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return s


def collect_data(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    session = get_session()
    scrapers: List[BaseScraper] = [
        FootyStatsScraper(session),
        Scores24Scraper(session),
    ]
    all_records: List[MatchRecord] = []
    source_counts: Dict[str, int] = {}
    for scraper in scrapers:
        try:
            rows = scraper.fetch(start_date, end_date)
            source_counts[scraper.source_name] = len(rows)
            all_records.extend(rows)
        except Exception as exc:
            source_counts[scraper.source_name] = 0
            print(f"[WARN] {scraper.source_name} failed: {exc}")
    print("[INFO] Source row counts:", ", ".join(f"{k}={v}" for k, v in source_counts.items()))
    weak_sources = [k for k, v in source_counts.items() if v == 0]
    if weak_sources:
        print(f"[WARN] Sources with no usable rows in this run: {', '.join(sorted(weak_sources))}")
    deduped = deduplicate_records(all_records)
    return records_to_df(deduped)


def merge_history_data(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "source",
        "league",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "odds_1x",
        "odds_x2",
    ]
    if existing.empty and fresh.empty:
        return pd.DataFrame(columns=base_cols)
    if existing.empty:
        merged = fresh.copy()
    elif fresh.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, fresh], ignore_index=True)

    merged = merged[base_cols].copy()
    merged["match_date"] = pd.to_datetime(merged["match_date"]).dt.date
    merged["home_team"] = merged["home_team"].fillna("").map(normalize_team_name)
    merged["away_team"] = merged["away_team"].fillna("").map(normalize_team_name)
    merged["league"] = merged["league"].fillna("unknown")
    merged["source"] = merged["source"].fillna("unknown")

    merged["quality_score"] = 0
    merged.loc[merged["odds_1x"].notna() | merged["odds_x2"].notna(), "quality_score"] += 1
    merged.loc[merged["home_goals"].notna() & merged["away_goals"].notna(), "quality_score"] += 2
    merged = merged.sort_values("quality_score", ascending=False)
    merged = merged.drop_duplicates(
        subset=["match_date", "home_team", "away_team"], keep="first"
    ).drop(columns=["quality_score"])
    return merged.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)


def load_history_store(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(
            columns=[
                "source",
                "league",
                "match_date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "odds_1x",
                "odds_x2",
            ]
        )
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    return df


def save_history_store(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def ensure_history_range(
    store_path: str,
    need_start: dt.date,
    need_end: dt.date,
) -> pd.DataFrame:
    existing = load_history_store(store_path)
    if not existing.empty and "match_date" in existing.columns:
        have_start = existing["match_date"].min()
        have_end = existing["match_date"].max()
    else:
        have_start = None
        have_end = None

    fetch_ranges: List[Tuple[dt.date, dt.date]] = []
    if have_start is None or have_end is None:
        fetch_ranges.append((need_start, need_end))
    else:
        if need_start < have_start:
            fetch_ranges.append((need_start, have_start - dt.timedelta(days=1)))
        if need_end > have_end:
            fetch_ranges.append((have_end + dt.timedelta(days=1), need_end))

    merged = existing.copy()
    for start, end in fetch_ranges:
        if start > end:
            continue
        print(f"Fetching missing history: {start.isoformat()} to {end.isoformat()} ...")
        fresh = collect_data(start, end)
        merged = merge_history_data(merged, fresh)

    if fetch_ranges:
        save_history_store(store_path, merged)
        print(f"History store updated: {store_path} ({len(merged)} rows)")
    return merged


def evaluate_prediction_result(market: str, home_goals: Optional[int], away_goals: Optional[int]) -> Optional[bool]:
    if home_goals is None or away_goals is None:
        return None
    if market == "1X":
        return home_goals >= away_goals
    if market == "X2":
        return away_goals >= home_goals
    return None


def pick_top_predictions_for_day(
    model: DixonColesModel,
    day_matches: pd.DataFrame,
    odds_target: float = 1.20,
    odds_tolerance: float = 0.05,
    top_n: int = 5,
) -> List[Prediction]:
    picks: List[Prediction] = []
    for _, row in day_matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        try:
            p_1x, p_x2 = model.predict_1x_x2(home, away)
        except KeyError:
            continue

        candidates = [
            ("1X", p_1x, row.get("odds_1x")),
            ("X2", p_x2, row.get("odds_x2")),
        ]
        for market, prob, offered in candidates:
            fair = 1.0 / prob if prob > 0 else np.inf
            edge = None
            if offered is not None and not pd.isna(offered):
                edge = (prob * offered) - 1.0
                if abs(offered - odds_target) > odds_tolerance:
                    continue
            else:
                # If bookmaker odds are missing in source data, select around target by model fair odds.
                if abs(fair - odds_target) > odds_tolerance:
                    continue
            result = evaluate_prediction_result(market, row.get("home_goals"), row.get("away_goals"))
            picks.append(
                Prediction(
                    match_date=row["match_date"],
                    league=row["league"],
                    home_team=home,
                    away_team=away,
                    market=market,
                    model_probability=float(prob),
                    fair_odds=float(fair),
                    offered_odds=None if offered is None or pd.isna(offered) else float(offered),
                    edge=None if edge is None else float(edge),
                    result=result,
                )
            )

    picks.sort(
        key=lambda p: (
            -p.model_probability,
            -(p.edge if p.edge is not None else -999.0),
        )
    )
    return picks[:top_n]


def run_backtest(
    df: pd.DataFrame,
    start_date: dt.date,
    days: int,
    train_lookback_days: int = 180,
    decay_xi: float = 0.0,
    odds_target: float = 1.20,
    odds_tolerance: float = 0.05,
    top_n: int = 5,
) -> pd.DataFrame:
    results = []
    for i in range(days):
        day = start_date + dt.timedelta(days=i)
        train_start = day - dt.timedelta(days=train_lookback_days)

        train = df[
            (df["match_date"] >= train_start)
            & (df["match_date"] < day)
            & df["home_goals"].notna()
            & df["away_goals"].notna()
        ].copy()
        test = df[df["match_date"] == day].copy()
        if train.empty or test.empty:
            continue

        common_teams = set(train["home_team"]).union(set(train["away_team"]))
        test = test[test["home_team"].isin(common_teams) & test["away_team"].isin(common_teams)]
        if test.empty:
            continue

        day_candidates: List[Prediction] = []
        for league in sorted(test["league"].dropna().unique()):
            train_l = train[train["league"] == league].copy()
            test_l = test[test["league"] == league].copy()
            if train_l.empty or test_l.empty:
                continue
            n_teams = len(set(train_l["home_team"]).union(set(train_l["away_team"])))
            if n_teams < 6 or len(train_l) < 35:
                continue

            model = DixonColesModel(max_goals=10)
            try:
                model.fit(
                    train_l[["home_team", "away_team", "home_goals", "away_goals", "match_date"]],
                    decay_xi=decay_xi,
                    reference_date=day - dt.timedelta(days=1),
                )
            except Exception:
                continue

            picks = pick_top_predictions_for_day(
                model=model,
                day_matches=test_l,
                odds_target=odds_target,
                odds_tolerance=odds_tolerance,
                top_n=top_n,
            )
            day_candidates.extend(picks)

        day_candidates.sort(
            key=lambda p: (
                -p.model_probability,
                -(p.edge if p.edge is not None else -999.0),
            )
        )
        for p in day_candidates[:top_n]:
            results.append(
                {
                    "match_date": p.match_date,
                    "league": p.league,
                    "home_team": p.home_team,
                    "away_team": p.away_team,
                    "market": p.market,
                    "model_probability": p.model_probability,
                    "fair_odds": p.fair_odds,
                    "offered_odds": p.offered_odds,
                    "edge": p.edge,
                    "won": p.result,
                }
            )
    return pd.DataFrame(results)


def predict_for_date(
    history_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    predict_date: dt.date,
    train_lookback_days: int = 180,
    decay_xi: float = 0.0,
    odds_target: float = 1.20,
    odds_tolerance: float = 0.05,
    top_n: int = 5,
    min_teams_per_league: int = DEFAULT_MIN_TEAMS_PER_LEAGUE,
    min_matches_per_league: int = DEFAULT_MIN_MATCHES_PER_LEAGUE,
) -> pd.DataFrame:
    out_cols = [
        "match_date",
        "league",
        "home_team",
        "away_team",
        "market",
        "model_probability",
        "fair_odds",
        "offered_odds",
        "edge",
    ]
    train_start = predict_date - dt.timedelta(days=train_lookback_days)
    train = history_df[
        (history_df["match_date"] >= train_start)
        & (history_df["match_date"] < predict_date)
        & history_df["home_goals"].notna()
        & history_df["away_goals"].notna()
    ].copy()
    test = fixtures_df[fixtures_df["match_date"] == predict_date].copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=out_cols)

    picks: List[Prediction] = []
    for league in sorted(test["league"].dropna().unique()):
        train_l = train[train["league"] == league].copy()
        test_l = test[test["league"] == league].copy()
        if train_l.empty or test_l.empty:
            continue
        n_teams = len(set(train_l["home_team"]).union(set(train_l["away_team"])))
        if n_teams < min_teams_per_league or len(train_l) < min_matches_per_league:
            continue
        model = DixonColesModel(max_goals=10)
        try:
            model.fit(
                train_l[["home_team", "away_team", "home_goals", "away_goals", "match_date"]],
                decay_xi=decay_xi,
                reference_date=predict_date - dt.timedelta(days=1),
            )
        except Exception:
            continue
        picks.extend(
            pick_top_predictions_for_day(
                model=model,
                day_matches=test_l,
                odds_target=odds_target,
                odds_tolerance=odds_tolerance,
                top_n=top_n,
            )
        )

    picks.sort(key=lambda p: -p.model_probability)
    rows = []
    for p in picks[:top_n]:
        rows.append(
            {
                "match_date": p.match_date,
                "league": p.league,
                "home_team": p.home_team,
                "away_team": p.away_team,
                "market": p.market,
                "model_probability": p.model_probability,
                "fair_odds": p.fair_odds,
                "offered_odds": p.offered_odds,
                "edge": p.edge,
            }
        )
    return pd.DataFrame(rows, columns=out_cols)


def print_summary(backtest_df: pd.DataFrame) -> None:
    if backtest_df.empty:
        print("No backtest predictions generated. Check source parsers and data availability.")
        return
    settled = backtest_df[backtest_df["won"].notna()].copy()
    n_total = len(backtest_df)
    n_settled = len(settled)
    n_won = int(settled["won"].sum()) if n_settled > 0 else 0
    hit_rate = (n_won / n_settled) if n_settled > 0 else float("nan")
    print(f"Predictions: {n_total}")
    print(f"Settled: {n_settled}")
    print(f"Wins: {n_won}")
    print(f"Hit rate: {hit_rate:.4f}" if n_settled else "Hit rate: n/a")

    if "offered_odds" in settled.columns and settled["offered_odds"].notna().any():
        pnl = np.where(settled["won"], settled["offered_odds"] - 1.0, -1.0).sum()
        roi = pnl / n_settled if n_settled else float("nan")
        print(f"Unit PnL: {pnl:.4f}")
        print(f"ROI: {roi:.4f}")


def compute_backtest_metrics(backtest_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    if backtest_df.empty:
        return {
            "predictions": 0,
            "settled": 0,
            "wins": 0,
            "hit_rate": float("nan"),
            "unit_pnl": float("nan"),
            "roi": float("nan"),
        }

    settled = backtest_df[backtest_df["won"].notna()].copy()
    n_total = len(backtest_df)
    n_settled = len(settled)
    n_won = int(settled["won"].sum()) if n_settled > 0 else 0
    hit_rate = (n_won / n_settled) if n_settled > 0 else float("nan")

    unit_pnl = float("nan")
    roi = float("nan")
    if n_settled > 0 and "offered_odds" in settled.columns and settled["offered_odds"].notna().any():
        unit_pnl = float(np.where(settled["won"], settled["offered_odds"] - 1.0, -1.0).sum())
        roi = unit_pnl / n_settled

    return {
        "predictions": int(n_total),
        "settled": int(n_settled),
        "wins": int(n_won),
        "hit_rate": float(hit_rate),
        "unit_pnl": float(unit_pnl),
        "roi": float(roi),
    }


def _sanitize_label(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


def resolve_backtest_output_path(
    requested_output: str,
    history_store: str,
    start_date: dt.date,
    end_date: dt.date,
    days: int,
) -> str:
    os.makedirs(SIMULATIONS_DIR, exist_ok=True)
    norm = requested_output.strip()
    requested_dir = os.path.dirname(norm)
    requested_base = os.path.basename(norm)
    default_name = "backtest_predictions.csv"

    if requested_dir:
        os.makedirs(requested_dir, exist_ok=True)
        return norm

    source = _sanitize_label(os.path.splitext(os.path.basename(history_store))[0]) or "history"
    auto_name = (
        f"backtest_{days}d_{source}_{start_date.isoformat()}_to_{end_date.isoformat()}.csv"
    )
    if requested_base == default_name:
        return os.path.join(SIMULATIONS_DIR, auto_name)
    return os.path.join(SIMULATIONS_DIR, requested_base)


def append_simulation_log(
    history_store: str,
    output_csv: str,
    start_date: dt.date,
    end_date: dt.date,
    args: argparse.Namespace,
    backtest_df: pd.DataFrame,
) -> str:
    os.makedirs(SIMULATIONS_DIR, exist_ok=True)
    log_path = os.path.join(SIMULATIONS_DIR, SIMULATION_LOG_FILE)
    metrics = compute_backtest_metrics(backtest_df)
    row = {
        "run_utc": dt.datetime.now(dt.UTC).isoformat(),
        "history_store": history_store,
        "output_csv": output_csv,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "days": int(args.days),
        "history_days": int(args.history_days),
        "lookback_days": int(args.lookback_days),
        "decay_xi": float(args.decay_xi),
        "top_n": int(args.top_n),
        "odds_target": float(args.odds_target),
        "odds_tolerance": float(args.odds_tolerance),
        "min_teams_per_league": int(args.min_teams_per_league),
        "min_matches_per_league": int(args.min_matches_per_league),
        "predictions": metrics["predictions"],
        "settled": metrics["settled"],
        "wins": metrics["wins"],
        "hit_rate": metrics["hit_rate"],
        "unit_pnl": metrics["unit_pnl"],
        "roi": metrics["roi"],
    }
    log_df = pd.DataFrame([row])
    if os.path.exists(log_path):
        prev = pd.read_csv(log_path)
        log_df = pd.concat([prev, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dixon-Coles 1X/X2 predictor using footystats.org, scores24.live and "
            "statschecker.com data, with daily top-5 picks and backtesting."
        )
    )
    parser.add_argument(
        "--history-store",
        default=HISTORY_STORE_DEFAULT,
        help=f"Local historical data store CSV. Default: {HISTORY_STORE_DEFAULT}.",
    )
    parser.add_argument(
        "--backfill-start-date",
        default=None,
        help="Optional backfill start date (YYYY-MM-DD) for the local history store.",
    )
    parser.add_argument(
        "--backfill-end-date",
        default=None,
        help="Optional backfill end date (YYYY-MM-DD) for the local history store.",
    )
    parser.add_argument(
        "--predict-date",
        default=None,
        help="Generate live picks for a specific date (YYYY-MM-DD), no settlement required.",
    )
    parser.add_argument(
        "--start-date",
        default="2026-01-01",
        help="Backtest start date in YYYY-MM-DD format. Default: 2026-01-01.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to backtest. Default: 30.",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=220,
        help="How many days of raw data to collect before start-date. Default: 220.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=180,
        help="Training lookback window for each day. Default: 180.",
    )
    parser.add_argument(
        "--decay-xi",
        type=float,
        default=0.0,
        help="Exponential time-decay rate per day for training weights. Default: 0.0 (disabled).",
    )
    parser.add_argument(
        "--odds-target",
        type=float,
        default=1.20,
        help="Target bookmaker odds for 1X/X2 picks. Default: 1.20.",
    )
    parser.add_argument(
        "--odds-tolerance",
        type=float,
        default=0.05,
        help="Allowed odds deviation from target. Default: 0.05 (=> 1.15-1.25 around 1.20).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Top predictions per day. Default: 5.",
    )
    parser.add_argument(
        "--output-csv",
        default="backtest_predictions.csv",
        help=(
            "Output CSV for generated picks. If only a filename is provided, "
            "backtest outputs are saved in the simulations/ folder."
        ),
    )
    parser.add_argument(
        "--min-teams-per-league",
        type=int,
        default=DEFAULT_MIN_TEAMS_PER_LEAGUE,
        help="Minimum number of teams in league history to fit model for live predictions.",
    )
    parser.add_argument(
        "--min-matches-per-league",
        type=int,
        default=DEFAULT_MIN_MATCHES_PER_LEAGUE,
        help="Minimum number of historical matches in league to fit model for live predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backfill_start_date and args.backfill_end_date:
        b_start = dt.date.fromisoformat(args.backfill_start_date)
        b_end = dt.date.fromisoformat(args.backfill_end_date)
        if b_start > b_end:
            raise ValueError("backfill-start-date must be <= backfill-end-date")
        df_hist = ensure_history_range(
            store_path=args.history_store,
            need_start=b_start,
            need_end=b_end,
        )
        print(
            f"Backfill complete for {b_start.isoformat()} to {b_end.isoformat()}. "
            f"Rows in store: {len(df_hist)}"
        )
        return

    if args.predict_date:
        predict_date = dt.date.fromisoformat(args.predict_date)
        collect_start = predict_date - dt.timedelta(days=args.history_days)
        train_end = predict_date - dt.timedelta(days=1)
        print(
            f"Ensuring training history from {collect_start.isoformat()} to "
            f"{train_end.isoformat()} ..."
        )
        history_df = ensure_history_range(
            store_path=args.history_store,
            need_start=collect_start,
            need_end=train_end,
        )
        history_df = history_df[
            (history_df["match_date"] >= collect_start)
            & (history_df["match_date"] <= train_end)
        ].copy()
        session = get_session()
        scores_scraper = Scores24Scraper(session)
        fixtures = scores_scraper.fetch(predict_date, predict_date)
        fixtures_df = records_to_df(deduplicate_records(fixtures))
        pred_df = predict_for_date(
            history_df=history_df,
            fixtures_df=fixtures_df,
            predict_date=predict_date,
            train_lookback_days=args.lookback_days,
            decay_xi=args.decay_xi,
            odds_target=args.odds_target,
            odds_tolerance=args.odds_tolerance,
            top_n=args.top_n,
            min_teams_per_league=args.min_teams_per_league,
            min_matches_per_league=args.min_matches_per_league,
        )
        if pred_df.empty:
            print("No predictions generated for requested date. Try increasing history-days.")
        else:
            print(f"Predictions generated: {len(pred_df)}")
            print(pred_df.to_string(index=False))
        pred_df.to_csv(args.output_csv, index=False)
        print(f"Saved picks to {args.output_csv}")
        return

    start_date = dt.date.fromisoformat(args.start_date)
    end_date = start_date + dt.timedelta(days=args.days - 1)
    collect_start = start_date - dt.timedelta(days=args.history_days)

    print(
        f"Ensuring history from {collect_start.isoformat()} to {end_date.isoformat()} "
        "from footystats.org, scores24.live ..."
    )
    df = ensure_history_range(
        store_path=args.history_store,
        need_start=collect_start,
        need_end=end_date,
    )
    df = df[(df["match_date"] >= collect_start) & (df["match_date"] <= end_date)].copy()
    if df.empty:
        print("No data collected. Update source parsers or use pre-exported source dumps.")
        return

    backtest_df = run_backtest(
        df=df,
        start_date=start_date,
        days=args.days,
        train_lookback_days=args.lookback_days,
        decay_xi=args.decay_xi,
        odds_target=args.odds_target,
        odds_tolerance=args.odds_tolerance,
        top_n=args.top_n,
    )
    print_summary(backtest_df)
    resolved_output = resolve_backtest_output_path(
        requested_output=args.output_csv,
        history_store=args.history_store,
        start_date=start_date,
        end_date=end_date,
        days=args.days,
    )
    backtest_df.to_csv(resolved_output, index=False)
    print(f"Saved picks to {resolved_output}")
    log_path = append_simulation_log(
        history_store=args.history_store,
        output_csv=resolved_output,
        start_date=start_date,
        end_date=end_date,
        args=args,
        backtest_df=backtest_df,
    )
    print(f"Simulation log updated: {log_path}")


if __name__ == "__main__":
    main()
