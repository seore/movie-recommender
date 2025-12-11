from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"

CONTENT_VECTORIZER_PATH = MODELS_DIR / "content_vectorizer.pkl"
ENCODERS_PATH = MODELS_DIR / "movie_id_encoder.pkl"
USER_ITEM_MATRIX_PATH = MODELS_DIR / "user_item_matrix.npz"


def train_content(movies: pd.DataFrame):
    if "genres" not in movies.columns:
        movies["genres"] = ""

    text_parts = [movies["genres"].fillna("")]
    if "overview" in movies.columns:
        text_parts.append(movies["overview"].fillna(""))
    elif "description" in movies.columns:
        text_parts.append(movies["description"].fillna(""))
    else:
        text_parts.append(movies["title"].fillna(""))

    combined = text_parts[0].astype(str) + " " + text_parts[1].astype(str)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(combined)

    payload = {
        "vectorizer": vectorizer,
        "movie_ids": movies["movieId"].tolist(),
    }

    with open(CONTENT_VECTORIZER_PATH, "wb") as f:
        pickle.dump(payload, f)

    print("Saved content_vectorizer.pkl")
    print("Content matrix shape:", matrix.shape)


def train_collab(ratings: pd.DataFrame):
    required = {"userId", "movieId", "rating"}
    if not required.issubset(ratings.columns):
        raise ValueError(f"ratings.csv must contain {required}")

    user_enc = LabelEncoder()
    movie_enc = LabelEncoder()

    ratings = ratings.copy()
    ratings["user_idx"] = user_enc.fit_transform(ratings["userId"])
    ratings["movie_idx"] = movie_enc.fit_transform(ratings["movieId"])

    num_users = ratings["user_idx"].nunique()
    num_movies = ratings["movie_idx"].nunique()
    print("Users:", num_users, "Movies:", num_movies)

    user_item = sp.coo_matrix(
        (ratings["rating"], (ratings["user_idx"], ratings["movie_idx"])),
        shape=(num_users, num_movies),
    ).tocsr()

    sp.save_npz(USER_ITEM_MATRIX_PATH, user_item)
    print("Saved user_item_matrix.npz")

    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump({"user_enc": user_enc, "movie_enc": movie_enc}, f)
    print("Saved movie_id_encoder.pkl")


def main():
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    print("Training content model...")
    train_content(movies)

    print("Training collaborative model...")
    train_collab(ratings)

    print("✅ All models trained.")


if __name__ == "__main__":
    main()
