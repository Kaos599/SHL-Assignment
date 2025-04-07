import streamlit as st
import requests
import pandas as pd
import json
import time

# Set page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; }
    .sub-header { font-size: 1.5rem; color: #3B82F6; margin-bottom: 2rem; }
    .recommendation-card { padding: 1.5rem; border-radius: 0.5rem; background-color: #F3F4F6; margin-bottom: 1rem; }
    .recommendation-title { font-size: 1.2rem; font-weight: bold; color: #1E3A8A; }
    .recommendation-detail { margin-top: 0.5rem; color: #4B5563; }
    .yes-badge { background-color: #10B981; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem; }
    .no-badge { background-color: #F87171; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem; }
    .score-badge { background-color: #3B82F6; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem; }
</style>""", unsafe_allow_html=True)

# API endpoint
API_BASE_URL = "http://localhost:8000"  # Update for deployment

def get_recommendations(query, url=None, max_results=10):
    """Get recommendations from the API."""
    try:
        params = {"query": query, "max_results": max_results}
        if url:
            params["url"] = url
            
        response = requests.get(f"{API_BASE_URL}/recommend", params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">SHL Assessment Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Find the perfect assessments for your hiring needs</div>', unsafe_allow_html=True)
    
    # Sidebar for information
    with st.sidebar:
        st.subheader("About")
        st.write("""This tool helps hiring managers find the most relevant SHL assessments for their job openings. Enter your job description or requirements, and we'll recommend the best assessments for your needs.""")
        
        st.subheader("How it works")
        st.write("""1. Enter a job description or requirements. 2. Optionally provide a job posting URL. 3. Our AI analyzes your input and matches it with SHL assessments. 4. Review recommended assessments and their details.""")
        
        st.subheader("Example queries")
        st.markdown("""- I am hiring for Java developers who can collaborate effectively with business teams. - Looking for an assessment for Python, SQL, and JavaScript skills under 60 minutes. - Need a cognitive assessment for a data analyst position.""")
    
    # Input section
    st.subheader("What are you looking for?")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area("Enter job description or requirements:", height=150, placeholder="E.g., I'm hiring for Java developers who can collaborate effectively with business teams.")
    
    with col2:
        url = st.text_input("Job posting URL (optional):", placeholder="https://example.com/job-posting")
        max_results = st.slider("Maximum results:", min_value=1, max_value=10, value=5)
    
    # Submit button
    if st.button("Get Recommendations", type="primary"):
        if not query:
            st.warning("Please enter a job description or requirements.")
            return
        
        with st.spinner("Finding the best assessments for you..."):
            start_time = time.time()
            results = get_recommendations(query, url, max_results)
            processing_time = time.time() - start_time
        
        if results:
            st.success(f"Found {len(results['recommendations'])} recommendations in {processing_time:.2f} seconds!")
            
            # Display enhanced query used
            with st.expander("See how we interpreted your request"):
                st.write(results.get('enhanced_query', 'No enhanced query available'))
            
            # Display recommendations
            st.subheader("Recommended Assessments")
            
            for rec in results['recommendations']:
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <div class="recommendation-title">{rec['name']}</div>
                        <div class="recommendation-detail">
                            <strong>Test Type:</strong> {rec['test_type']} | 
                            <strong>Duration:</strong> {rec['duration']} | 
                            <span class="{'yes-badge' if rec['remote_testing'] == 'Yes' else 'no-badge'}">Remote Testing: {rec['remote_testing']}</span> | 
                            <span class="{'yes-badge' if rec['adaptive_irt'] == 'Yes' else 'no-badge'}">Adaptive: {rec['adaptive_irt']}</span> | 
                            <span class="score-badge">Relevance: {rec['score']}</span>
                        </div>
                        <div class="recommendation-detail">
                            <a href="{rec['url']}" target="_blank">View Assessment Details</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show as a dataframe for easy comparison
            with st.expander("View as table"):
                df = pd.DataFrame(results['recommendations'])
                df = df[['name', 'test_type', 'duration', 'remote_testing', 'adaptive_irt', 'score']]
                df.columns = ['Assessment', 'Test Type', 'Duration', 'Remote Testing', 'Adaptive/IRT', 'Relevance Score']
                st.dataframe(df)

if __name__ == "__main__":
    main()