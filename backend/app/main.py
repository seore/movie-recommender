import os
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .schemas import (
    RecommendRequest,
    RecommendResponse,
    Recommendation,
    MovieListResponse,
    MovieSummary,
)
from .recommender import engine

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_BASE_URL = "https://api.themoviedb.org/3"


app = FastAPI(
    title="Cine Recommender API",
    description="Hybrid movie recommendation system (content + collaborative filtering)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for dev; restrict in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    try:
        engine.init()
    except Exception as e:
        print(f"[ERROR] Failed to initialise engine: {e}")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.get("/movies", response_model=MovieListResponse, tags=["movies"])
def get_movies(limit: int = 20):
    try:
        movies = engine.list_sample_movies(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return MovieListResponse(
        movies=[
            MovieSummary(
                movie_id=m["movie_id"],
                title=m["title"],
                genres=m.get("genres"),
            )
            for m in movies
        ]
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(payload: RecommendRequest):
    try:
        result = engine.recommend(
            movie_id=payload.movie_id,
            top_k=payload.top_k,
            alpha=payload.alpha,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Missing file: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not result.recommendations:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for movie_id={payload.movie_id}.",
        )

    return RecommendResponse(
        base_movie_id=result.base_movie_id,
        base_title=result.base_title,
        recommendations=[
            Recommendation(
                movie_id=r["movie_id"],
                title=r["title"],
                score=r["score"],
            )
            for r in result.recommendations
        ],
    )

@app.get("/poster", tags=["poster"])
async def get_poster(title: str, year: int | None = None):
    if not TMDB_API_KEY:
        return {"poster_url": None}

    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
    }
    if year:
        params["year"] = year

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                f"{TMDB_BASE_URL}/search/movie",
                params=params,
                timeout=10.0,
            )
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(f"[ERROR] TMDB request failed: {e}")
        return {"poster_url": None}

    results = data.get("results") or []
    if not results:
        return {"poster_url": None}

    poster_path = results[0].get("poster_path")
    if not poster_path:
        return {"poster_url": None}

    return {"poster_url": f"{TMDB_IMG_BASE}{poster_path}"}
