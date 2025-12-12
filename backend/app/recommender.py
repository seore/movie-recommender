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
        self._load_ncf_embeddings()

        self._initialized = True

    def recommend(
        self,
        movie_id: int,
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_ALPHA,
        mode: str = "hybrid",
    ) -> RecommendationResult:
        """
        mode:
          - "content": only content-based similarity
          - "collab": only collaborative similarity
          - "hybrid": weighted combination
        """
        if not self._initialized:
            self.init()

        if top_k < 1:
            top_k = 1

        content_scores = self._content_scores_for_movie(movie_id)
        collab_scores = self._collab_scores_for_movie(movie_id)

        content_norm = normalize_scores(content_scores)
        collab_norm = normalize_scores(collab_scores)

        mode = (mode or "hybrid").lower()
        if mode not in {"content", "collab", "hybrid"}:
            mode = "hybrid"

        if mode == "content":
            final_scores = content_norm
        elif mode == "collab":
            final_scores = collab_norm
        else:
            alpha = float(min(max(alpha, 0.0), 1.0))
            final_scores = alpha * collab_norm + (1.0 - alpha) * content_norm

        base_row = self._row_idx_from_movie_id(movie_id)
        if base_row is not None:
            final_scores[base_row] = -np.inf  # exclude the base movies

        top_indices = np.argsort(final_scores)[::-1][:top_k]
        scores = final_scores[top_indices]

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
    
    def explore_movies(
        self,
        page: int = 1,
        page_size: int = 20,
        genre_substring: str | None = None,
        min_rating: float | None = None,
    ) -> dict:
        if not self._initialized:
            self.init()

        df = self.movies.copy()

        if genre_substring:
            g = genre_substring.lower()
            if "genres" in df.columns:
                df = df[df["genres"].str.lower().str.contains(g, na=False)]

        if min_rating is not None and "vote_average" in df.columns:
            df = df[df["vote_average"].fillna(0) >= float(min_rating)]

        total = len(df)
        page = max(1, page)
        page_size = max(1, min(page_size, 100))

        start = (page - 1) * page_size
        end = start + page_size
        page_df = df.iloc[start:end]

        movies = []
        for _, row in page_df.iterrows():
            movies.append(
                {
                    "movie_id": int(row["movieId"]),
                    "title": str(row["title"]),
                    "genres": None if "genres" not in row or pd.isna(row["genres"]) else str(row["genres"]),
                    "vote_average": float(row["vote_average"]) if "vote_average" in df.columns and not pd.isna(row["vote_average"]) else None,
                }
            )

        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "movies": movies,
        }


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
        SBERT embeddings. If not then TF-IDF.
        """
        sbert_path = MODELS_DIR / "sbert_embeddings.npz"
        if sbert_path.exists():
            print("[CONTENT] Using SBERT embeddings")
            data = np.load(sbert_path)
            self.sbert_embeddings = data["embeddings"]
            self.use_sbert = True
            return
        
        # --- TF-IDF fallback ---
        print("[CONTENT] SBERT not found, falling back to TF-IDF")
        from sklearn.feature_extraction.text import TfidfVectorizer

        text_parts = [self.movies["genres"].fillna("")]
        if "overview" in self.movies.columns:
            text_parts.append(self.movies["overview"].fillna(""))
        elif "description" in self.movies.columns:
            text_parts.append(self.movies["description"].fillna(""))
        else:
            text_parts.append(self.movies["title"].fillna(""))

        combined = text_parts[0].astype(str) + " " + text_parts[1].astype(str)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.content_matrix = self.vectorizer.fit_transform(combined)
        self.use_sbert = False

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

        self.item_similarity = cosine_similarity(self.user_item_matrix.T)

    def _load_ncf_embeddings(self) -> None:
        path = MODELS_DIR / "ncf_item_embeddings.npz"
        if not path.exists():
            print("[COLLAB] NCF embeddings no found, use CF instead.")
            self.ncf_embeddings = None
            return
        
        data = np.load(path)
        self.ncf_movie_ids = data["movie_ids"].astype(int)
        self.ncf_embeddings = data["embeddings"]
        self.ncf_movieid_to_idx = {
            int(mid): idx for idx, mid in enumerate(self.ncf_movie_ids)
        }

        print("[COLLAB] Loaded NCF item embeddings:", self.ncf_embeddings.shape)


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
        
        # SBERT path
        if getattr(self, "use_sbert", False) and hasattr(self, "sbert_embeddings"):
            if row_idx >= self.sbert_embeddings.shape[0]:
                return np.zeros(self.movies.shape[0])
            vec = self.sbert_embeddings[row_idx : row_idx + 1]  # (1, d)
            sim = cosine_similarity(vec, self.sbert_embeddings)[0]
            return sim

        # TF-IDF fallback
        movie_vec = self.content_matrix[row_idx]
        sim = cosine_similarity(movie_vec, self.content_matrix)[0]
        return sim

    def _collab_scores_for_movie(self, movie_id: int) -> np.ndarray:
        # Prefer NCF embeddings if available
        if getattr(self, "ncf_embeddings", None) is not None:
            idx = getattr(self, "ncf_movieid_to_idx", {}).get(int(movie_id))
            if idx is None:
                return np.zeros(self.movies.shape[0])

            vec = self.ncf_embeddings[idx : idx + 1]  # (1, d)
            sims = cosine_similarity(vec, self.ncf_embeddings)[0]  # (num_items,)

            scores_aligned = np.zeros(self.movies.shape[0])
            for mid, score in zip(self.ncf_movie_ids, sims):
                row_idx = self._row_idx_from_movie_id(int(mid))
                if row_idx is not None:
                    scores_aligned[row_idx] = score
            return scores_aligned

        # Fallback: traditional item-item similarity from ratings
        movie_idx = self._get_encoded_idx_from_movie_id(movie_id)
        if movie_idx is None or movie_idx >= self.item_similarity.shape[0]:
            return np.zeros(self.movies.shape[0])

        sim_scores = self.item_similarity[movie_idx]
        encoded_indices = np.arange(len(sim_scores))
        movie_ids = self.movie_enc.inverse_transform(encoded_indices)

        scores_aligned = np.zeros(self.movies.shape[0])
        for enc_idx, score in zip(encoded_indices, sim_scores):
            mid = int(movie_ids[enc_idx])
            row_idx = self._row_idx_from_movie_id(mid)
            if row_idx is not None:
                scores_aligned[row_idx] = score
        return scores_aligned
    
    # movie search
    def search_movies(self, query: str, limit: int = 20) -> list[dict]:
        if not query:
            return self.list_sample_movies(limit)
        q = query.strip().lower()
        if not q:
            return self.list_sample_movies(limit)

        mask = self.movies["title"].str.lower().str.contains(q, na=False)
        results = self.movies[mask].head(limit)[["movieId", "title", "genres"]]
        return [
            {
                "movie_id": int(row["movieId"]),
                "title": str(row["title"]),
                "genres": None if pd.isna(row["genres"]) else str(row["genres"]),
            }
            for _, row in results.iterrows()
        ]


# singleton engine used by main.py
engine = RecommenderEngine()