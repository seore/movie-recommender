# backend/prepare_movielens.py

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_movielens(url: str = MOVIELENS_URL) -> bytes:
    print(f"Downloading MovieLens dataset from {url} ...")
    resp = requests.get(url)
    resp.raise_for_status()
    print("Download complete.")
    return resp.content


def extract_csv_from_zip(zip_bytes: bytes, member_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with zf.open(member_name) as f:
            if member_name.endswith("movies.csv"):
                df = pd.read_csv(f)
            elif member_name.endswith("ratings.csv"):
                df = pd.read_csv(f)
            else:
                raise ValueError(f"Unknown file type: {member_name}")
    return df


def main():
    zip_bytes = download_movielens()

    # The paths inside the ZIP for ml-latest-small
    movies_member = "ml-latest-small/movies.csv"
    ratings_member = "ml-latest-small/ratings.csv"

    print("Extracting movies...")
    movies_df = extract_csv_from_zip(zip_bytes, movies_member)
    print("Extracting ratings...")
    ratings_df = extract_csv_from_zip(zip_bytes, ratings_member)

    # Sanity check / light cleaning
    required_movie_cols = {"movieId", "title", "genres"}
    required_rating_cols = {"userId", "movieId", "rating"}

    if not required_movie_cols.issubset(movies_df.columns):
        raise ValueError(
            f"movies.csv is missing required columns {required_movie_cols}, got {movies_df.columns}"
        )

    if not required_rating_cols.issubset(ratings_df.columns):
        raise ValueError(
            f"ratings.csv is missing required columns {required_rating_cols}, got {ratings_df.columns}"
        )

    # Save into backend/data/
    movies_out = DATA_DIR / "movies.csv"
    ratings_out = DATA_DIR / "ratings.csv"

    print(f"Writing {movies_out} ...")
    movies_df.to_csv(movies_out, index=False)
    print(f"Writing {ratings_out} ...")
    ratings_df.to_csv(ratings_out, index=False)

    print("Done. Sample rows:")
    print(movies_df.head())
    print(ratings_df.head())


if __name__ == "__main__":
    main()
