from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
from dotenv import load_dotenv

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  

TMDB_API_KEY = os.getenv("TMDB_API_KEY")


def ensure_tmdb_key() -> str:
    if not TMDB_API_KEY:
        raise RuntimeError(
            "TMDB_API_KEY is not set. Add it to backend/.env as TMDB_API_KEY=..."
        )
    return TMDB_API_KEY


def split_title_year(full_title: str) -> tuple[str, Optional[str]]:
    """
    Extract title + year from strings like 'Star Trek V: The Final Frontier (1989)'.
    """
    import re

    s = str(full_title).strip()
    m = re.match(r"^(.*)\s+\((\d{4})\)$", s)
    if not m:
        return s, None
    return m.group(1), m.group(2)


def tmdb_search_movie(title: str, year: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Search TMDB for a movie by title (and optional year).
    Returns the best match or None.
    """
    api_key = ensure_tmdb_key()

    params = {
        "api_key": api_key,
        "query": title,
    }
    if year:
        params["year"] = year

    try:
        res = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(f"[WARN] TMDB search failed for '{title}' ({year}): {e}")
        return None

    results = data.get("results") or []
    if not results:
        return None

    return results[0]


def tmdb_fetch_list(endpoint: str, pages: int = 5, delay: float = 0.25) -> List[Dict[str, Any]]:
    """
    Fetch multiple pages of TMDB movies from a given 'list-like' endpoint,
    e.g. 'movie/popular' or 'movie/top_rated'.
    """
    api_key = ensure_tmdb_key()
    all_results: List[Dict[str, Any]] = []

    for page in range(1, pages + 1):
        params = {"api_key": api_key, "page": page}
        url = f"{TMDB_BASE_URL}/{endpoint}"
        try:
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            print(f"[WARN] TMDB fetch failed for {endpoint} page {page}: {e}")
            break

        results = data.get("results") or []
        if not results:
            break

        all_results.extend(results)
        print(f"[TMDB] Fetched {endpoint} page {page} ({len(results)} movies), total={len(all_results)}")
        time.sleep(delay)

    return all_results
