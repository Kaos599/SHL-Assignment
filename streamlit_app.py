import streamlit as st
import pandas as pd
import json
import time
import os
from dotenv import load_dotenv
import requests
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API endpoint - Allow switching between local and production with environment variable
API_ENDPOINT = "http://localhost:8000"

# If API_ENDPOINT doesn't include http(s)://, add it
if not API_ENDPOINT.startswith("http"):
    API_ENDPOINT = f"http://{API_ENDPOINT}"

logger.info(f"Using API endpoint: {API_ENDPOINT}")

# Set page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #262730;
        color: #FFFFFF;
    }
    .main-header { 
        font-size: 2.5rem; 
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header { 
        font-size: 1.5rem; 
        color: #66B2FF; 
        text-align: center;
        margin-bottom: 2rem;
    }
    .author-info {
        text-align: center;
        color: #A0AEC0;
        margin-bottom: 2rem;
    }
    .notice-box {
        background-color: #2D3748;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ECC94B;
    }
    .recommendation-card { 
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        background-color: #323742; 
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
    }
    .recommendation-title { 
        font-size: 1.2rem; 
        font-weight: bold; 
        color: #FFFFFF; 
        margin-bottom: 0.5rem;
    }
    .recommendation-detail { 
        margin-top: 0.5rem; 
        color: #D1D5DB;
        line-height: 1.5;
    }
    .yes-badge { 
        background-color: #10B981; 
        color: white; 
        padding: 0.25rem 0.5rem; 
        border-radius: 0.25rem; 
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .no-badge { 
        background-color: #F87171; 
        color: white; 
        padding: 0.25rem 0.5rem; 
        border-radius: 0.25rem; 
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .score-badge { 
        background-color: #3B82F6; 
        color: white; 
        padding: 0.25rem 0.5rem; 
        border-radius: 0.25rem; 
        font-size: 0.8rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4299E1;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3182CE;
    }
</style>
""", unsafe_allow_html=True)

def get_recommendations_from_api(query, url=None, max_results=10):
    """Get recommendations by calling the API endpoint."""
    try:
        # Prepare the request data
        payload = {
            "query": query,
            "max_results": max_results
        }
        
        if url:
            payload["url"] = url
            
        # Make the API request
        logger.info(f"Sending request to {API_ENDPOINT}/recommend")
        response = requests.post(f"{API_ENDPOINT}/recommend", json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Return the JSON response
        return response.json()
    except Exception as e:
        logger.error(f"Error getting recommendations from API: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Error getting recommendations: {str(e)}")
        return None

def main():
    # Header and Author Information
    st.markdown('<div class="main-header">SHL Assessment Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">For AI Research Intern Position in SHL</div>', unsafe_allow_html=True)
    st.markdown('<div class="author-info">Made by Harsh Dayal<br>(harshdayal13@gmail.com)</div>', unsafe_allow_html=True)
    
    # Notice about API loading time
    st.markdown("""
    <div class="notice-box">
        <strong>⚠️ Notice:</strong> I am using Render's free tier to deploy my API so the API might take a minute or two to load up before it can start receiving requests. Please be patient.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for information
    with st.sidebar:
        st.subheader("About")
        st.write("""
        This tool helps hiring managers find the most relevant SHL assessments for their job openings. 
        Enter your job description or requirements, and we'll recommend the best assessments for your needs.
        """)
        
        st.subheader("How it works")
        st.write("""
        1. Enter a job description or requirements
        2. Optionally provide a job posting URL
        3. Our AI analyzes your input and matches it with SHL assessments
        4. Review recommended assessments and their details
        """)
        
        st.subheader("Example queries")
        st.markdown("""
        - I am hiring for Java developers who can collaborate effectively with business teams
        - Looking for an assessment for Python, SQL, and JavaScript skills under 60 minutes
        - Need a cognitive assessment for a data analyst position
        """)
        
        # Display API endpoint (helpful for debugging)
        st.subheader("Settings")
        st.text(f"API: {API_ENDPOINT}")
    
    # Input section
    st.subheader("What are you looking for?")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Enter job description or requirements:",
            height=150,
            placeholder="E.g., I'm hiring for Java developers who can collaborate effectively with business teams."
        )
    
    with col2:
        url = st.text_input(
            "Job posting URL (optional):",
            placeholder="https://example.com/job-posting"
        )
        max_results = st.slider(
            "Maximum results:",
            min_value=1,
            max_value=10,
            value=5
        )
    
    # Submit button
    if st.button("Get Recommendations", type="primary"):
        if not query:
            st.warning("Please enter a job description or requirements.")
            return
        
        with st.spinner("Finding the best assessments for you..."):
            start_time = time.time()
            
            # Get recommendations from API
            results = get_recommendations_from_api(query, url, max_results)
            processing_time = time.time() - start_time
        
        if results:
            st.success(f"Found {len(results['recommendations'])} recommendations in {processing_time:.2f} seconds!")
            
            # Display enhanced query used
            if results.get('enhanced_query'):
                with st.expander("See how we interpreted your request"):
                    st.write(results.get('enhanced_query', 'No enhanced query available'))
            
            # Display recommendations
            st.subheader("Recommended Assessments")
            
            # Display natural language response if available
            if results.get('natural_language_response'):
                st.markdown("### Analysis")
                st.markdown(
                    f'<div style="background-color: #404452; padding: 15px; border-radius: 10px; margin-bottom: 20px; color: #D1D5DB;">{results["natural_language_response"]}</div>',
                    unsafe_allow_html=True
                )
            
            for rec in results['recommendations']:
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <div class="recommendation-title">{rec.get('name', 'Unknown')}</div>
                        <div class="recommendation-detail">
                            <strong>Test Type:</strong> {rec.get('test_type', 'Unknown')} | 
                            <strong>Duration:</strong> {rec.get('duration_minutes', 'Unknown')} minutes | 
                            <span class="{'yes-badge' if rec.get('remote_testing') == 'Yes' else 'no-badge'}">Remote Testing: {rec.get('remote_testing', 'No')}</span> | 
                            <span class="{'yes-badge' if rec.get('adaptive_irt') == 'Yes' else 'no-badge'}">Adaptive: {rec.get('adaptive_irt', 'No')}</span> | 
                            <span class="score-badge">Relevance: {rec.get('score', 0)}</span>
                        </div>
                        <div class="recommendation-detail">
                            <a href="{rec.get('url', '#')}" target="_blank" style="color: #4299E1; text-decoration: none;">View Assessment Details →</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show as a dataframe for easy comparison
            with st.expander("View as table"):
                df = pd.DataFrame(results['recommendations'])
                columns_to_show = ['name', 'test_type', 'duration_minutes', 'remote_testing', 'adaptive_irt', 'score']
                columns_to_show = [col for col in columns_to_show if col in df.columns]
                df = df[columns_to_show]
                column_names = {
                    'name': 'Assessment', 
                    'test_type': 'Test Type', 
                    'duration_minutes': 'Duration (minutes)', 
                    'remote_testing': 'Remote Testing', 
                    'adaptive_irt': 'Adaptive/IRT', 
                    'score': 'Relevance Score'
                }
                df.columns = [column_names.get(col, col) for col in df.columns]
                st.dataframe(df)

if __name__ == "__main__":
    main()