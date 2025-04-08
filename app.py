from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any, Union
import uvicorn
from recommendation_engine import SHLRecommendationEngine
import os
from dotenv import load_dotenv
import logging
import traceback
from bson import ObjectId
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Custom JSON encoder to handle ObjectId
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

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

recommendation_engine = None

def get_recommendation_engine():
    """Get or create the recommendation engine lazily."""
    global recommendation_engine
    if recommendation_engine is None:
        try:
            logger.info("Initializing recommendation engine...")
            recommendation_engine = SHLRecommendationEngine(skip_embedding_creation=True)
            logger.info("Recommendation engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize recommendation engine: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize recommendation engine: {str(e)}")
    return recommendation_engine

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
    natural_language_response: Optional[str] = None

    class Config:
        json_encoders = {
            ObjectId: str
        }

# Root route
@app.get("/", tags=["General"])
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "SHL Assessment Recommendation API is running"}

# Middleware to log request details
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request path: {request.url.path}")
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

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
        logger.info(f"Processing recommendation request with query: {query}")
        
        # Get recommendations using lazy-loaded engine
        engine = get_recommendation_engine()
        recommendations = engine.recommend(query, url, max_results)
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        # Log the first recommendation to debug ObjectId serialization
        if recommendations:
            logger.info(f"First recommendation structure: {json.dumps(recommendations[0], cls=CustomJSONEncoder)}")
        
        # Extract the enhanced query if available
        enhanced_query = None
        duration_constraint = engine.extract_duration_constraint(query)
        logger.info(f"Duration constraint: {duration_constraint}")
        
        # Generate natural language response using Gemini
        natural_language_response = engine.generate_natural_language_response(
            query, recommendations, duration_constraint
        )
        logger.info(f"Generated natural language response: {natural_language_response}")
        
        # Convert ObjectId to string in recommendations
        for rec in recommendations:
            if '_id' in rec:
                rec['_id'] = str(rec['_id'])
        
        return {
            "query": query,
            "enhanced_query": enhanced_query,
            "recommendations": recommendations,
            "duration_constraint": duration_constraint,
            "natural_language_response": natural_language_response
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# POST endpoint for recommendations
@app.post("/recommend", tags=["Recommendations"], response_model=RecommendationResponse)
async def post_recommendations(request: RecommendationRequest):
    """
    Get SHL assessment recommendations using a POST request with JSON body.
    Provide a natural language query or job description, and optionally a URL to a job posting.
    """
    try:
        logger.info(f"Processing POST recommendation request with query: {request.query}")
        
        # Get recommendations using lazy-loaded engine
        engine = get_recommendation_engine()
        recommendations = engine.recommend(
            request.query, 
            str(request.url) if request.url else None, 
            request.max_results
        )
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        # Log the first recommendation to debug ObjectId serialization
        if recommendations:
            logger.info(f"First recommendation structure: {json.dumps(recommendations[0], cls=CustomJSONEncoder)}")
        
        # Extract the enhanced query if available
        enhanced_query = None
        duration_constraint = engine.extract_duration_constraint(request.query)
        logger.info(f"Duration constraint: {duration_constraint}")
        
        # Generate natural language response using Gemini
        natural_language_response = engine.generate_natural_language_response(
            request.query, recommendations, duration_constraint
        )
        logger.info(f"Generated natural language response: {natural_language_response}")
        
        # Convert ObjectId to string in recommendations
        for rec in recommendations:
            if '_id' in rec:
                rec['_id'] = str(rec['_id'])
        
        return {
            "query": request.query,
            "enhanced_query": enhanced_query,
            "recommendations": recommendations,
            "duration_constraint": duration_constraint,
            "natural_language_response": natural_language_response
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 8000))
    
    # Run the FastAPI app
    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)