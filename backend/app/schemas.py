from pydantic import BaseModel, Field
from typing import List


class RecommendRequest(BaseModel):
    movie_id: int = Field(..., description="Movie ID to base recommendations on")
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations")
    alpha: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Hybrid weight: 1.0 = collaborative only, 0.0 = content only",
    )


class Recommendation(BaseModel):
    movie_id: int
    title: str
    score: float


class RecommendResponse(BaseModel):
    base_movie_id: int
    base_title: str
    recommendations: List[Recommendation]


class MovieSummary(BaseModel):
    movie_id: int
    title: str
    genres: str | None = None


class MovieListResponse(BaseModel):
    movies: List[MovieSummary]
