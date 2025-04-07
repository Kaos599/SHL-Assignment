from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any, Union
import uvicorn
from recommendation_engine import SHLRecommendationEngine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API for recommending SHL assessments based on job descriptions and queries",
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
    max_results: Optional[int] = 10

class RecommendationResponse(BaseModel):
    query: str
    enhanced_query: Optional[str] = None
    recommendations: List[Dict[str, Any]]
    duration_constraint: Optional[int] = None

# Root route
@app.get("/", tags=["General"])
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "SHL Assessment Recommendation API is running"}

# GET endpoint for recommendations
@app.get("/recommend", tags=["Recommendations"], response_model=RecommendationResponse)
async def get_recommendations(
    query: str = Query(..., description="Natural language query or job description"),
    url: Optional[str] = Query(None, description="URL to a job posting"),
    max_results: Optional[int] = Query(10, description="Maximum number of results to return", ge=1, le=20)
):
    """
    Get SHL assessment recommendations based on a natural language query or job description.
    Optionally provide a URL to a job posting for additional context.
    """
    try:
        # Get recommendations
        recommendations = recommendation_engine.recommend(query, url, max_results)
        
        # Extract the enhanced query if available
        enhanced_query = None
        duration_constraint = recommendation_engine.extract_duration_constraint(query)
        
        return {
            "query": query,
            "enhanced_query": enhanced_query,
            "recommendations": recommendations,
            "duration_constraint": duration_constraint
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# POST endpoint for recommendations
@app.post("/recommend", tags=["Recommendations"], response_model=RecommendationResponse)
async def post_recommendations(request: RecommendationRequest):
    """
    Get SHL assessment recommendations using a POST request with JSON body.
    Provide a natural language query or job description, and optionally a URL to a job posting.
    """
    try:
        # Get recommendations
        recommendations = recommendation_engine.recommend(
            request.query, 
            str(request.url) if request.url else None, 
            request.max_results
        )
        
        # Extract the enhanced query if available
        enhanced_query = None
        duration_constraint = recommendation_engine.extract_duration_constraint(request.query)
        
        return {
            "query": request.query,
            "enhanced_query": enhanced_query,
            "recommendations": recommendations,
            "duration_constraint": duration_constraint
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 8000))
    
    # Run the FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)