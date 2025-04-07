import os
import pandas as pd
import numpy as np
from recommendation_engine import SHLRecommendationEngine
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

def load_test_queries():
    """Load test queries for evaluation"""
    test_queries = [
        {
            "id": "q1",
            "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
        },
        {
            "id": "q2",
            "query": "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript. Need an assessment package that can test all skills with max duration of 60 minutes."
        },
        {
            "id": "q3",
            "query": "I am hiring for an analyst and want to screen applications using Cognitive and personality tests. What options are available within 45 minutes?"
        },
        {
            "id": "q4", 
            "query": "We need to hire customer service representatives. Looking for a package that assesses interpersonal skills and problem-solving ability."
        },
        {
            "id": "q5",
            "query": "Need to assess leadership potential for executive candidates. Prefer assessments that are adaptive and can be taken remotely."
        }
    ]
    return test_queries

def run_test():
    """Test the recommendation engine with sample queries"""
    
    logger.info("Initializing SHL Recommendation Engine...")
    engine = SHLRecommendationEngine()
    
    logger.info("Loading test queries...")
    test_queries = load_test_queries()
    
    for i, query_data in enumerate(test_queries):
        query_id = query_data["id"]
        query = query_data["query"]
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing Query {query_id}: {query}")
        
        # Get recommendations
        recommendations = engine.recommend(query, top_k=5)
        
        logger.info(f"\nTop 5 Recommendations for Query {query_id}:\n")
        
        for j, rec in enumerate(recommendations, 1):
            # Display recommendation details
            test_types = ", ".join(rec.get('test_type_names', [])) if 'test_type_names' in rec else ", ".join(rec.get('test_type', []))
            logger.info(f"{j}. {rec['name']} (Score: {rec['score']:.4f})")
            logger.info(f"   Test Types: {test_types}")
            logger.info(f"   Duration: {rec.get('duration', 'Unknown')}")
            logger.info(f"   Remote Testing: {rec.get('remote_testing', False)}, Adaptive: {rec.get('adaptive_irt', False)}")
            logger.info(f"   URL: {rec.get('url', '#')}")
            logger.info("")

def main():
    """Main function to run the test"""
    logger.info("Starting SHL Assessment Recommendation Test")
    
    try:
        run_test()
        logger.info("Test completed successfully!")
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    main() 