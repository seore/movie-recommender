from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"

CONTENT_VECTORIZER_PATH = MODELS_DIR / "content_vectorizer.pkl"
ENCODERS_PATH = MODELS_DIR / "movie_id_encoder.pkl"
USER_ITEM_MATRIX_PATH = MODELS_DIR / "user_item_matrix.npz"

# Hybrid weight: alpha=1.0 -> pure collaborative; 0.0 -> pure content
DEFAULT_ALPHA = 0.5
DEFAULT_TOP_K = 10