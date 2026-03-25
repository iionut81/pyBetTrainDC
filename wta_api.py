from __future__ import annotations

"""
wta_api.py
HTTP session with retries + optional JSON disk cache for WTA API (and similar GET JSON).
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_json_session(
    user_agent: str,
    *,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_json(
    session: requests.Session,
    url: str,
    *,
    timeout: float = 20.0,
    verify: bool = True,
    cache_dir: Optional[Path] = None,
    cache_ttl_seconds: float = 0.0,
) -> Any:
    """GET JSON with optional TTL cache on disk (cache disabled if ttl <= 0 or cache_dir is None)."""
    if cache_dir is not None and cache_ttl_seconds > 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:48]
        path = cache_dir / f"{key}.json"
        if path.is_file():
            age = time.time() - path.stat().st_mtime
            if age < cache_ttl_seconds:
                return json.loads(path.read_text(encoding="utf-8"))

    resp = session.get(url, timeout=timeout, verify=verify)
    resp.raise_for_status()
    data = resp.json()

    if cache_dir is not None and cache_ttl_seconds > 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:48]
        path = cache_dir / f"{key}.json"
        path.write_text(json.dumps(data), encoding="utf-8")

    return data
