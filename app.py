from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
import uvicorn
import time
from recommendation_engine import SHLRecommendationEngine

# Initialize the FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API for recommending SHL assessments based on job descriptions",
    version="1.0.0"
)

# Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the recommendation engine
recommendation_engine = SHLRecommendationEngine()

# Define request and response models
class RecommendationRequest(BaseModel):
    query: str
    url: Optional[HttpUrl] = None
    max_results: int = 10

class Assessment(BaseModel):
    name: str
    url: str
    remote_testing: str
    adaptive_irt: str
    duration: str
    test_type: str
    score: float

class RecommendationResponse(BaseModel):
    recommendations: List[Assessment]
    query: str
    enhanced_query: Optional[str] = None
    processing_time: float

# Root route
@app.get("/")
async def root():
    return {"message": "SHL Assessment Recommendation API is running"}

# POST endpoint for recommendations
@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: RecommendationRequest):
    start_time = time.time()
    
    if not request.query and not request.url:
        raise HTTPException(status_code=400, detail="Either query or URL must be provided")
    
    enhanced_query = recommendation_engine.enhance_query(request.query)
    
    recommendations = recommendation_engine.recommend(
        query=request.query,
        url=request.url.url if request.url else None,
        top_k=min(request.max_results, 10)
    )
    
    processing_time = time.time() - start_time
    
    return RecommendationResponse(
        recommendations=recommendations,
        query=request.query,
        enhanced_query=enhanced_query,
        processing_time=round(processing_time, 3)
    )

# GET endpoint for recommendations
@app.get("/recommend", response_model=RecommendationResponse)
async def recommend_assessments_get(
    query: str = Query(..., description="Natural language query or job description"),
    url: Optional[str] = Query(None, description="URL of job description"),
    max_results: int = Query(10, description="Maximum number of results to return")
):
    start_time = time.time()
    
    enhanced_query = recommendation_engine.enhance_query(query)
    
    recommendations = recommendation_engine.recommend(
        query=query,
        url=url,
        top_k=min(max_results, 10)
    )
    
    processing_time = time.time() - start_time
    
    return RecommendationResponse(
        recommendations=recommendations,
        query=query,
        enhanced_query=enhanced_query,
        processing_time=round(processing_time, 3)
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)