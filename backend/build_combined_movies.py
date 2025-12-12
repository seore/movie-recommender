from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Set

import pandas as pd

from tmdb_utils import split_title_year, tmdb_search_movie, tmdb_fetch_list

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MOVIES_IN_PATH = DATA_DIR / "movies.csv"
MOVIES_COMBINED_PATH = DATA_DIR / "movies_combined.csv"


def enrich_movielens_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """
    For each MovieLens movie, attempt to find a TMDB match and enrich with metadata.
    """
    print(f"Enriching {len(movies)} MovieLens movies with TMDB metadata...")

    for col in ["tmdb_id", "overview", "release_date", "vote_average", "vote_count", "poster_path"]:
        if col not in movies.columns:
            movies[col] = pd.NA

    for idx, row in movies.iterrows():
        full_title = row["title"]
        title, year = split_title_year(full_title)
        print(f"[ML {idx+1}/{len(movies)}] {full_title} -> TMDB search", end="\r")

        result = tmdb_search_movie(title, year)
        if not result:
            continue

        movies.at[idx, "tmdb_id"] = result.get("id")
        movies.at[idx, "overview"] = result.get("overview")
        movies.at[idx, "release_date"] = result.get("release_date")
        movies.at[idx, "vote_average"] = result.get("vote_average")
        movies.at[idx, "vote_count"] = result.get("vote_count")
        movies.at[idx, "poster_path"] = result.get("poster_path")

    print("\nDone enriching MovieLens movies.")
    return movies


def fetch_extra_tmdb_movies(pages_popular: int = 10, pages_top_rated: int = 10) -> pd.DataFrame:
    """
    Fetch additional movies from TMDB (popular + top-rated).
    Returns a DataFrame with one row per TMDB movie.
    """
    print("Fetching extra TMDB movies (popular + top_rated)...")
    popular = tmdb_fetch_list("movie/popular", pages=pages_popular)
    top_rated = tmdb_fetch_list("movie/top_rated", pages=pages_top_rated)

    all_movies = {m["id"]: m for m in (popular + top_rated)}  
    print(f"Fetched total extra TMDB unique movies: {len(all_movies)}")

    rows = []
    for tmdb_id, m in all_movies.items():
        rows.append(
            {
                "tmdb_id": tmdb_id,
                "title": m.get("title"),
                "original_title": m.get("original_title"),
                "overview": m.get("overview"),
                "release_date": m.get("release_date"),
                "vote_average": m.get("vote_average"),
                "vote_count": m.get("vote_count"),
                "poster_path": m.get("poster_path"),
            }
        )

    df = pd.DataFrame(rows)
    print("Sample TMDB-only movies:")
    print(df.head())
    return df


def build_combined_movies():
    if not MOVIES_IN_PATH.exists():
        raise FileNotFoundError(f"No MovieLens movies.csv found at {MOVIES_IN_PATH}")

    movies_ml = pd.read_csv(MOVIES_IN_PATH)
    print(f"Loaded {len(movies_ml)} MovieLens movies.")

    movies_ml = enrich_movielens_movies(movies_ml)

    tmdb_extra = fetch_extra_tmdb_movies(pages_popular=10, pages_top_rated=10)
    tmdb_extra = tmdb_extra.dropna(subset=["title"]).reset_index(drop=True)

    ml_tmdb_ids: Set[int] = set(
        int(x) for x in movies_ml["tmdb_id"].dropna().astype(int).tolist()
    )

    tmdb_only = tmdb_extra[~tmdb_extra["tmdb_id"].isin(ml_tmdb_ids)].copy()
    print(f"TMDB-only movies after excluding ML overlaps: {len(tmdb_only)}")

    max_movie_id = int(movies_ml["movieId"].max()) if len(movies_ml) else 0
    new_ids = list(range(max_movie_id + 1, max_movie_id + 1 + len(tmdb_only)))
    tmdb_only["movieId"] = new_ids

    if "genres" not in movies_ml.columns:
        movies_ml["genres"] = ""
    tmdb_only["genres"] = ""

    common_cols = list(movies_ml.columns)
    for col in ["tmdb_id", "overview", "release_date", "vote_average", "vote_count", "poster_path"]:
        if col not in common_cols:
            common_cols.append(col)

    for col in common_cols:
        if col not in movies_ml.columns:
            movies_ml[col] = pd.NA
        if col not in tmdb_only.columns:
            tmdb_only[col] = pd.NA

    movies_ml = movies_ml[common_cols]
    tmdb_only = tmdb_only[common_cols]

    combined = pd.concat([movies_ml, tmdb_only], ignore_index=True)
    print(f"Combined dataset size: {len(combined)} movies")

    combined.to_csv(MOVIES_COMBINED_PATH, index=False)
    print(f"Saved combined movies -> {MOVIES_COMBINED_PATH}")
    print("Preview:")
    print(combined.head())


if __name__ == "__main__":
    build_combined_movies()
