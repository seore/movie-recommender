from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    RecommendRequest,
    RecommendResponse,
    Recommendation,
    MovieListResponse,
    MovieSummary,
)
from .recommender import engine


app = FastAPI(
    title="Cine Recommender API",
    description="Hybrid movie recommendation system (content + collaborative filtering)",
    version="1.0.0",
)

# Allow all origins for dev; tighten this in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """
    Initialise the recommender on startup so first request is fast.
    """
    try:
        engine.init()
    except Exception as e:
        # You can log here with logging module
        print(f"[ERROR] Failed to initialise recommender: {e}")


@app.get("/health", tags=["meta"])
def health_check():
    return {"status": "ok"}


@app.get("/movies", response_model=MovieListResponse, tags=["movies"])
def list_movies(limit: int = 20):
    """
    Return a small random set of movies for the frontend to show
    as examples / dropdown options.
    """
    try:
        engine.init()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    movies = engine.list_sample_movies(limit)
    return MovieListResponse(
        movies=[
            MovieSummary(
                movie_id=m["movie_id"], title=m["title"], genres=m.get("genres")
            )
            for m in movies
        ]
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend_movies(payload: RecommendRequest):
    """
    Recommend movies given a base movie ID, using a hybrid of content-based
    similarity and collaborative filtering.
    """
    try:
        engine.init()
        result = engine.recommend(
            movie_id=payload.movie_id,
            top_k=payload.top_k,
            alpha=payload.alpha,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=500, detail=f"Data file missing: {fe}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not result.recommendations:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for movie_id={payload.movie_id}. "
                   f"It might not exist in the dataset or has no similar movies.",
        )

    return RecommendResponse(
        base_movie_id=result.base_movie_id,
        base_title=result.base_title,
        recommendations=[
            Recommendation(
                movie_id=rec["movie_id"],
                title=rec["title"],
                score=rec["score"],
            )
            for rec in result.recommendations
        ],
    )
