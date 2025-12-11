from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz, csr_matrix

from .utils import normalize_scores, load_csv_with_required_columns


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MOVIES_PATH = DATA_DIR / "movies.csv"
CONTENT_VECTORIZER_PATH = MODELS_DIR / "content_vectorizer.pkl"
ENCODERS_PATH = MODELS_DIR / "movie_id_encoder.pkl"
USER_ITEM_MATRIX_PATH = MODELS_DIR / "user_item_matrix.npz"

DEFAULT_TOP_K = 10
DEFAULT_ALPHA = 0.5


@dataclass
class RecommendationResult:
    base_movie_id: int
    base_title: str
    recommendations: List[Dict]


class RecommenderEngine:
    """
    Hybrid movie recommender:
    - Content-based: TF-IDF similarities between movie descriptions
      (genres + overview/description/title)
    - Collaborative: item-based similarity over a user-item rating matrix.
    """

    def __init__(self) -> None:
        self._initialized = False

    # ---------- PUBLIC API ----------

    def init(self) -> None:
        """Load data + model artefacts once on startup."""
        if self._initialized:
            return

        self._load_movies()
        self._load_content_model()
        self._load_collaborative_models()

        self._initialized = True

    def recommend(
        self,
        movie_id: int,
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_ALPHA,
    ) -> RecommendationResult:
        """Return hybrid recommendations for a given base movie ID."""
        if not self._initialized:
            self.init()

        if top_k < 1:
            top_k = 1

        content_scores = self._content_scores_for_movie(movie_id)
        collab_scores = self._collab_scores_for_movie(movie_id)

        content_norm = normalize_scores(content_scores)
        collab_norm = normalize_scores(collab_scores)

        alpha = float(min(max(alpha, 0.0), 1.0))
        hybrid_scores = alpha * collab_norm + (1.0 - alpha) * content_norm

        base_row = self._row_idx_from_movie_id(movie_id)
        if base_row is not None:
            hybrid_scores[base_row] = -np.inf  # never recommend the base movie itself

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        scores = hybrid_scores[top_indices]

        recs: List[Dict] = []
        for idx, score in zip(top_indices, scores):
            if score <= 0:
                continue
            row = self.movies.iloc[idx]
            recs.append(
                {
                    "movie_id": int(row["movieId"]),
                    "title": str(row["title"]),
                    "score": float(round(float(score), 4)),
                }
            )

        base_title = self.get_base_movie_title(movie_id)

        return RecommendationResult(
            base_movie_id=int(movie_id),
            base_title=base_title,
            recommendations=recs,
        )

    def list_sample_movies(self, n: int = 20) -> List[Dict]:
        """Return a random subset of movies for dropdowns etc."""
        if not self._initialized:
            self.init()

        n = max(1, min(n, 100))
        sample = self.movies.sample(n, random_state=42)[["movieId", "title", "genres"]]
        return [
            {
                "movie_id": int(row["movieId"]),
                "title": str(row["title"]),
                "genres": None if pd.isna(row["genres"]) else str(row["genres"]),
            }
            for _, row in sample.iterrows()
        ]

    # ---------- LOADING ----------

    def _load_movies(self) -> None:
        self.movies = load_csv_with_required_columns(
            MOVIES_PATH,
            ["movieId", "title"],
            friendly_name="movies.csv",
        ).reset_index(drop=True)

        if "genres" not in self.movies.columns:
            self.movies["genres"] = ""

        # Map external movieId → row index
        self.movieid_to_row = {
            int(mid): idx for idx, mid in enumerate(self.movies["movieId"].tolist())
        }

    def _load_content_model(self) -> None:
        """
        Load TF-IDF vectorizer and build the content matrix for all movies
        using the same vocabulary that was fit during training.
        """
        with open(CONTENT_VECTORIZER_PATH, "rb") as f:
            payload = pickle.load(f)

        self.vectorizer: TfidfVectorizer = payload["vectorizer"]

        text_parts = [self.movies["genres"].fillna("")]
        if "overview" in self.movies.columns:
            text_parts.append(self.movies["overview"].fillna(""))
        elif "description" in self.movies.columns:
            text_parts.append(self.movies["description"].fillna(""))
        else:
            text_parts.append(self.movies["title"].fillna(""))

        combined = text_parts[0].astype(str) + " " + text_parts[1].astype(str)
        self.content_matrix = self.vectorizer.transform(combined)

    def _load_collaborative_models(self) -> None:
        """
        Load user-item matrix + encoders and compute item-item similarity.
        """
        self.user_item_matrix: csr_matrix = load_npz(
            USER_ITEM_MATRIX_PATH
        ).tocsr()

        with open(ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)

        self.user_enc = encoders["user_enc"]
        self.movie_enc = encoders["movie_enc"]

        # item-item similarity: movies x movies (in encoded index space)
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)

    # ---------- INTERNAL HELPERS ----------

    def _row_idx_from_movie_id(self, movie_id: int) -> Optional[int]:
        return self.movieid_to_row.get(int(movie_id))

    def _encoded_idx_from_movie_id(self, movie_id: int) -> Optional[int]:
        try:
            idx = self.movie_enc.transform([movie_id])[0]
            return int(idx)
        except Exception:
            return None

    def get_base_movie_title(self, movie_id: int) -> str:
        row_idx = self._row_idx_from_movie_id(movie_id)
        if row_idx is None:
            return f"Unknown movie ({movie_id})"
        return str(self.movies.iloc[row_idx]["title"])

    # ----- score computations -----

    def _content_scores_for_movie(self, movie_id: int) -> np.ndarray:
        row_idx = self._row_idx_from_movie_id(movie_id)
        if row_idx is None:
            return np.zeros(self.movies.shape[0])

        movie_vec = self.content_matrix[row_idx]
        sim = cosine_similarity(movie_vec, self.content_matrix)[0]
        return sim

    def _collab_scores_for_movie(self, movie_id: int) -> np.ndarray:
        enc_idx = self._encoded_idx_from_movie_id(movie_id)
        if enc_idx is None or enc_idx >= self.item_similarity.shape[0]:
            return np.zeros(self.movies.shape[0])

        sim_scores = self.item_similarity[enc_idx]  # length = num_movies_encoded

        encoded_indices = np.arange(len(sim_scores))
        movie_ids = self.movie_enc.inverse_transform(encoded_indices)

        scores_aligned = np.zeros(self.movies.shape[0])
        for idx_enc, score in zip(encoded_indices, sim_scores):
            mid = int(movie_ids[idx_enc])
            row_idx = self._row_idx_from_movie_id(mid)
            if row_idx is not None:
                scores_aligned[row_idx] = score

        return scores_aligned


# global singleton engine used by main.py
engine = RecommenderEngine()