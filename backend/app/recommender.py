from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix, csr_matrix

from .config import MOVIES_PATH, RATINGS_PATH, DEFAULT_ALPHA, DEFAULT_TOP_K


@dataclass
class RecommendationResult:
    base_movie_id: int
    base_title: str
    recommendations: List[Dict]


class RecommenderEngine:
    """
    Simple hybrid recommender:
    - Content-based: TF-IDF on genres + optional overview/description
    - Collaborative: item-based cosine similarity over user-item ratings
    """

    def __init__(self):
        self._initialized = False

    def _load_data(self):
        # Load movies
        movies = pd.read_csv(MOVIES_PATH)

        # Normalise column names a bit
        movies.columns = [c.strip() for c in movies.columns]

        # Ensure required columns exist
        required_movie_cols = {"movieId", "title"}
        if not required_movie_cols.issubset(movies.columns):
            raise ValueError(
                f"movies.csv must contain columns: {required_movie_cols}, found: {movies.columns}"
            )

        # Genres may be missing
        if "genres" not in movies.columns:
            movies["genres"] = ""

        self.movies = movies.reset_index(drop=True)

        # Build a mapping from movieId to index
        self.movieid_to_idx = {
            int(mid): idx for idx, mid in enumerate(self.movies["movieId"].tolist())
        }

        # Load ratings
        ratings = pd.read_csv(RATINGS_PATH)
        ratings.columns = [c.strip() for c in ratings.columns]

        required_rating_cols = {"userId", "movieId", "rating"}
        if not required_rating_cols.issubset(ratings.columns):
            raise ValueError(
                f"ratings.csv must contain columns: {required_rating_cols}, found: {ratings.columns}"
            )

        self.ratings = ratings

    def _build_content_matrix(self):
        # Use genres + optional overview/description as text
        text_parts = [self.movies["genres"].fillna("")]

        if "overview" in self.movies.columns:
            text_parts.append(self.movies["overview"].fillna(""))
        elif "description" in self.movies.columns:
            text_parts.append(self.movies["description"].fillna(""))
        else:
            # Use title as last resort (not ideal, but better than nothing)
            text_parts.append(self.movies["title"].fillna(""))

        combined_text = (" " + " ".join([s for s in [""]])).join(
            [""] * len(self.movies)
        )  # placeholder, will overwrite below
        combined_text = (
            text_parts[0].astype(str)
            + " "
            + (text_parts[1] if len(text_parts) > 1 else "")
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=5000
        )
        self.content_matrix = self.vectorizer.fit_transform(combined_text)

    def _build_collaborative_matrices(self):
        ratings = self.ratings

        # Encode user and movie IDs to contiguous indices
        self.user_enc = LabelEncoder()
        self.movie_enc = LabelEncoder()

        ratings["user_idx"] = self.user_enc.fit_transform(ratings["userId"])
        ratings["movie_idx"] = self.movie_enc.fit_transform(ratings["movieId"])

        num_users = ratings["user_idx"].nunique()
        num_movies = ratings["movie_idx"].nunique()

        self.user_item_matrix: csr_matrix = coo_matrix(
            (ratings["rating"], (ratings["user_idx"], ratings["movie_idx"])),
            shape=(num_users, num_movies),
        ).tocsr()

        # Compute item-item similarity (cosine) over columns (movies)
        # For MovieLens 100k, this is fine. For larger datasets, you’d want a more
        # scalable approach (e.g. approximate nearest neighbours).
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)

    def init(self):
        if self._initialized:
            return

        self._load_data()
        self._build_content_matrix()
        self._build_collaborative_matrices()

        self._initialized = True

    # ---------- Helper methods ----------

    def _get_movie_idx_from_id(self, movie_id: int) -> int | None:
        # For collaborative part, we use encoded indices (movie_enc)
        # MovieLens IDs likely not 0..N-1, so we need to transform
        try:
            idx = self.movie_enc.transform([movie_id])[0]
            return int(idx)
        except Exception:
            return None

    def _get_internal_row_index(self, movie_id: int) -> int | None:
        """Index into self.movies / content_matrix"""
        return self.movieid_to_idx.get(int(movie_id))

    def get_base_movie_title(self, movie_id: int) -> str:
        row_idx = self._get_internal_row_index(movie_id)
        if row_idx is None:
            return f"Unknown movie ({movie_id})"
        return str(self.movies.iloc[row_idx]["title"])

    # ---------- Content-based ----------

    def _content_scores_for_movie(self, movie_id: int) -> np.ndarray:
        row_idx = self._get_internal_row_index(movie_id)
        if row_idx is None:
            # Unknown movie in movies.csv
            return np.zeros(self.movies.shape[0])

        movie_vec = self.content_matrix[row_idx]
        sim_scores = cosine_similarity(movie_vec, self.content_matrix)[0]
        return sim_scores

    # ---------- Collaborative (item-based) ----------

    def _collab_scores_for_movie(self, movie_id: int) -> np.ndarray:
        movie_idx = self._get_movie_idx_from_id(movie_id)
        if movie_idx is None or movie_idx >= self.item_similarity.shape[0]:
            # Unknown movie or not present in ratings
            return np.zeros(self.item_similarity.shape[0])

        sim_scores = self.item_similarity[movie_idx]  # shape: (num_movies_encoded,)
        # Map from encoded movie_idx back to global movieId and then to row index in movies
        encoded_indices = np.arange(len(sim_scores))
        movie_ids_encoded = self.movie_enc.inverse_transform(encoded_indices)

        # Build an array aligned with self.movies rows
        # For movies that are not in ratings, score = 0
        scores_aligned = np.zeros(self.movies.shape[0])

        for enc_idx, sim in zip(encoded_indices, sim_scores):
            movie_id_val = int(self.movie_enc.inverse_transform([enc_idx])[0])
            row_idx = self._get_internal_row_index(movie_id_val)
            if row_idx is not None:
                scores_aligned[row_idx] = sim

        return scores_aligned

    # ---------- Hybrid Recommendation ----------

    def recommend(
        self,
        movie_id: int,
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_ALPHA,
    ) -> RecommendationResult:
        if not self._initialized:
            self.init()

        if top_k < 1:
            top_k = 1

        # Content scores
        content_scores = self._content_scores_for_movie(movie_id)

        # Collaborative scores
        collab_scores = self._collab_scores_for_movie(movie_id)

        # Normalise both to [0, 1] to combine
        def normalize(arr: np.ndarray) -> np.ndarray:
            arr = arr.copy()
            arr[arr < 0] = 0
            max_val = arr.max()
            if max_val == 0:
                return arr
            return arr / max_val

        content_norm = normalize(content_scores)
        collab_norm = normalize(collab_scores)

        # Hybrid
        alpha = float(alpha)
        if alpha < 0:
            alpha = 0.0
        if alpha > 1:
            alpha = 1.0

        hybrid_scores = alpha * collab_norm + (1.0 - alpha) * content_norm

        # Remove the base movie from candidates
        base_row_idx = self._get_internal_row_index(movie_id)
        if base_row_idx is not None:
            hybrid_scores[base_row_idx] = -np.inf

        # Get top_k indices
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        scores = hybrid_scores[top_indices]

        recs = []
        for idx, score in zip(top_indices, scores):
            if score <= 0:
                # No meaningful similarity – can break or skip
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
        

# Singleton engine instance
engine = RecommenderEngine()
