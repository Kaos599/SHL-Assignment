import streamlit as st
import pandas as pd
import json
import time
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import numpy as np
import faiss
import langchain_google_genai as genai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from bs4 import BeautifulSoup
import logging
import traceback
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as google_genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB connection settings
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "SHL")
MONGO_COLLECTION_PACKAGED = os.getenv("MONGO_COLLECTION_PACKAGED", "packaged_solutions")
MONGO_COLLECTION_INDIVIDUAL = os.getenv("MONGO_COLLECTION_INDIVIDUAL", "individual_solutions")
MONGO_EMBEDDINGS_COLLECTION = os.getenv("MONGO_EMBEDDINGS_COLLECTION", "embeddings")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Google API key for Gemini
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
google_genai.configure(api_key=GOOGLE_API_KEY)

# Set page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #262730;  /* backgroundColor */
        color: #FFFFFF;              /* textColor */
    }
    .main-header { font-size: 2.5rem; color: #FFFFFF; }
    .sub-header { font-size: 1.5rem; color: #66B2FF; margin-bottom: 2rem; }
    .recommendation-card { padding: 1.5rem; border-radius: 0.5rem; background-color: #323742; margin-bottom: 1rem; }
    .recommendation-title { font-size: 1.2rem; font-weight: bold; color: #FFFFFF; }
    .recommendation-detail { margin-top: 0.5rem; color: #D1D5DB; }
    .yes-badge { background-color: #10B981; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem; }
    .no-badge { background-color: #F87171; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem; }
    .score-badge { background-color: #3B82F6; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_mongo_client():
    """Create and cache MongoDB client."""
    try:
        client = MongoClient(MONGO_URI)
        logger.info("MongoDB client initialized")
        return client
    except Exception as e:
        logger.error(f"Error initializing MongoDB client: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Could not connect to MongoDB: {str(e)}")
        return None

@st.cache_data
def load_assessments_from_mongodb():
    """Load assessment data from MongoDB."""
    client = get_mongo_client()
    if not client:
        return None
    
    try:
        db = client[MONGO_DB]
        packaged_data = list(db[MONGO_COLLECTION_PACKAGED].find({}))
        individual_data = list(db[MONGO_COLLECTION_INDIVIDUAL].find({}))
        
        # Combine data
        data = packaged_data + individual_data
        
        logger.info(f"Loaded {len(data)} assessments from MongoDB")
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading assessments from MongoDB: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Error loading assessments: {str(e)}")
        return None
    finally:
        # Don't close client here as it's cached
        pass

@st.cache_data
def load_embeddings_from_mongodb():
    """Load embeddings from MongoDB."""
    client = get_mongo_client()
    if not client:
        return None, None, None
    
    try:
        db = client[MONGO_DB]
        embeddings_collection = db[MONGO_EMBEDDINGS_COLLECTION]
        
        # Get embeddings
        embeddings_data = list(embeddings_collection.find({}))
        
        # Create a mapping from assessment_id to embedding
        embedding_map = {item["assessment_id"]: item["embedding"] for item in embeddings_data if "embedding" in item}
        
        # Create a mapping from assessment_id to text
        text_map = {item["assessment_id"]: item.get("text", "") for item in embeddings_data if "assessment_id" in item}
        
        # Get embeddings array for vector search
        embeddings_list = [item["embedding"] for item in embeddings_data if "embedding" in item]
        
        if not embeddings_list:
            logger.error("No embeddings found in MongoDB")
            return None, None, None
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings_list).astype('float32')
        
        # Create FAISS index
        dimension = len(embeddings_list[0])
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings_array)
        
        logger.info(f"Loaded {len(embeddings_list)} embeddings from MongoDB with dimension {dimension}")
        
        # Return the index, mapping, and dimension
        return index, embedding_map, text_map
    except Exception as e:
        logger.error(f"Error loading embeddings from MongoDB: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Error loading embeddings: {str(e)}")
        return None, None, None

@st.cache_resource
def initialize_gemini():
    """Initialize Gemini language model."""
    try:
        gemini = genai.ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        logger.info("Gemini language model initialized")
        return gemini
    except Exception as e:
        logger.error(f"Error initializing Gemini: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Error initializing Gemini: {str(e)}")
        return None

@st.cache_resource
def get_gemini_embeddings():
    """Initialize and return Gemini embedding model."""
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        logger.info("Gemini embedding model initialized")
        return embedding_model
    except Exception as e:
        logger.error(f"Error initializing Gemini embedding model: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Error initializing Gemini embedding model: {str(e)}")
        return None

def get_query_embedding(text):
    """Get embedding for query text using Gemini."""
    embedding_model = get_gemini_embeddings()
    if not embedding_model:
        return None
        
    try:
        embedding = embedding_model.embed_query(text)
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding from Gemini: {e}")
        logger.error(traceback.format_exc())
        
        # Try direct API as fallback
        try:
            logger.info("Trying direct Gemini API as fallback")
            result = google_genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_query")
            embedding = result["embedding"]
            return embedding
        except Exception as fallback_error:
            logger.error(f"Fallback embedding also failed: {fallback_error}")
            return None

def extract_duration_constraint(query):
    """Extract duration constraint from the query."""
    time_patterns = [
        r'under (\d+)\s*(?:minute|min)',
        r'less than (\d+)\s*(?:minute|min)',
        r'shorter than (\d+)\s*(?:minute|min)',
        r'(\d+)\s*(?:minute|min) or less',
        r'maximum (?:of )?(\d+)\s*(?:minute|min)',
        r'no longer than (\d+)\s*(?:minute|min)',
        r'no more than (\d+)\s*(?:minute|min)',
        r'within (\d+)\s*(?:minute|min)',
        r'(\d+)\s*(?:minute|min) maximum',
        r'up to (\d+)\s*(?:minute|min)'
    ]
    
    import re
    for pattern in time_patterns:
        match = re.search(pattern, query.lower())
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
    return None

def fetch_content_from_url(url):
    """Fetch and parse content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from the job description
        text = ' '.join([p.get_text() for p in soup.find_all(['p', 'div', 'li', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
        return text
    except Exception as e:
        logger.error(f"Error fetching URL content: {e}")
        return None

def search_keywords(query_text, text_map, top_k=10):
    """Search for relevant assessments using TF-IDF and cosine similarity."""
    try:
        # Create corpus from text_map values
        corpus = list(text_map.values())
        assessment_ids = list(text_map.keys())
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Fit and transform the corpus
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Transform the query
        query_vector = vectorizer.transform([query_text])
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get top k indices
        top_indices = cosine_similarities.argsort()[-top_k:][::-1]
        
        # Create results with assessment_ids and scores
        results = [(assessment_ids[idx], float(cosine_similarities[idx])) for idx in top_indices]
        
        logger.info(f"Keyword search found {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        logger.error(traceback.format_exc())
        return []

def search_embeddings(user_query, url=None, top_k=10, duration_constraint=None):
    """Search for relevant assessments based on the query."""
    # Load necessary resources
    assessments_df = load_assessments_from_mongodb()
    index, embedding_map, text_map = load_embeddings_from_mongodb()
    gemini = initialize_gemini()
    
    if assessments_df is None or index is None or embedding_map is None or gemini is None:
        return None
    
    try:
        # Extract embeddings from query
        query_text = user_query
        
        # If URL is provided, fetch content and append to query
        if url:
            url_content = fetch_content_from_url(url)
            if url_content:
                query_text = f"{query_text}\n\nJob Posting Content:\n{url_content}"
        
        # Generate enhanced query using Gemini
        prompt_template = PromptTemplate.from_template(
            "You are a tool that extracts key skills, competencies, and requirements from job descriptions. "
            "Extract the most important keywords from the text below that would be relevant for selecting "
            "appropriate assessments or tests. Focus on technical skills, competencies, job roles, and requirements.\n\n"
            "Text: {query}\n\n"
            "Output only the key terms as a comma-separated list."
        )
        
        chain = prompt_template | gemini | StrOutputParser()
        enhanced_query = chain.invoke({"query": query_text})
        
        # Combine original query with enhanced keywords
        search_query = f"{user_query} {enhanced_query}"
        
        # Try vector search first, fall back to keyword search if dimensions don't match
        try:
            logger.info("Attempting vector search with embeddings")
            
            # Create embedding for search query using Gemini
            query_embedding = get_query_embedding(search_query)
            
            if query_embedding is None:
                logger.warning("Failed to create query embedding, falling back to keyword search")
                raise ValueError("Failed to create query embedding")
            
            # Get dimension of the index
            index_dimension = index.d
            query_dimension = len(query_embedding)
            
            logger.info(f"Index dimension: {index_dimension}, Query dimension: {query_dimension}")
            
            if query_dimension != index_dimension:
                logger.warning(f"Dimension mismatch: index={index_dimension}, query={query_dimension}")
                # Try resizing the embedding to match index dimension
                try:
                    if query_dimension > index_dimension:
                        # Truncate the embedding
                        query_embedding = query_embedding[:index_dimension]
                    else:
                        # Pad with zeros
                        padding = np.zeros(index_dimension - query_dimension)
                        query_embedding = np.concatenate([query_embedding, padding])
                        
                    logger.info(f"Resized query embedding to dimension {len(query_embedding)}")
                except Exception as resize_error:
                    logger.error(f"Error resizing embedding: {resize_error}")
                    raise ValueError("Dimension mismatch between query and index")
            
            # Convert to numpy array
            query_embedding_np = np.array([query_embedding]).astype('float32')
            
            # Search FAISS index
            scores, indices = index.search(query_embedding_np, min(top_k * 3, index.ntotal))
            
            # Get assessment IDs from indices
            assessment_ids = list(embedding_map.keys())
            
            # Create results list with (assessment_id, score) pairs
            vector_results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(assessment_ids):
                    continue
                    
                assessment_id = assessment_ids[idx]
                score = float(scores[0][i])
                vector_results.append((assessment_id, score))
                
            search_results = vector_results
            search_method = "vector"
            
        except Exception as e:
            logger.warning(f"Vector search failed: {e}. Falling back to keyword search.")
            
            # Use keyword search as fallback
            search_results = search_keywords(search_query, text_map, top_k * 3)
            search_method = "keyword"
        
        # Prepare final results
        results = []
        for assessment_id, score in search_results:
            # Find the corresponding assessment
            assessment_row = assessments_df[assessments_df['_id'].astype(str) == assessment_id]
            
            if assessment_row.empty:
                continue
                
            assessment = assessment_row.iloc[0].to_dict()
            
            # Check duration constraint if specified
            if duration_constraint is not None:
                try:
                    # Extract numeric part of duration
                    duration_str = assessment.get('duration', '')
                    import re
                    duration_match = re.search(r'(\d+)', duration_str)
                    if duration_match:
                        assessment_duration = int(duration_match.group(1))
                        if assessment_duration > duration_constraint:
                            continue  # Skip this assessment if it exceeds the time constraint
                except (ValueError, AttributeError):
                    pass  # If we can't parse the duration, include it anyway
            
            # Add score to the assessment
            assessment['score'] = round(score, 2)
            
            results.append(assessment)
            
            # Check if we have enough results
            if len(results) >= top_k:
                break
        
        # Generate natural language response
        natural_language_response = generate_natural_language_response(user_query, results, duration_constraint)
        
        return {
            "query": user_query,
            "enhanced_query": enhanced_query,
            "recommendations": results,
            "duration_constraint": duration_constraint,
            "natural_language_response": natural_language_response,
            "search_method": search_method
        }
    except Exception as e:
        logger.error(f"Error searching embeddings: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Error searching embeddings: {str(e)}")
        return None

def generate_natural_language_response(query, recommendations, duration_constraint=None):
    """Generate natural language response using Gemini."""
    gemini = initialize_gemini()
    if not gemini:
        return None
        
    try:
        # Convert recommendations to text
        rec_text = ""
        for i, rec in enumerate(recommendations[:5], 1):  # Include only top 5 in prompt
            rec_text += f"{i}. {rec.get('name', 'Unknown')} - Type: {rec.get('test_type', 'Unknown')}, Duration: {rec.get('duration', 'Unknown')}\n"
        
        # Create prompt for Gemini
        prompt_template = PromptTemplate.from_template(
            "You are an expert in SHL assessments and HR. A hiring manager needs help selecting the right assessments for a job. "
            "Based on the job requirements and the top recommended assessments, provide a helpful analysis explaining why these assessments "
            "are suitable and how they match the job requirements. Be concise but informative.\n\n"
            "Job requirements: {query}\n\n"
            "{duration_text}\n\n"
            "Top recommended assessments:\n{recommendations}\n\n"
            "Provide a brief, helpful explanation (no more than 150 words) of why these assessments are recommended for this job:"
        )
        
        # Format duration constraint
        duration_text = ""
        if duration_constraint:
            duration_text = f"Duration constraint: The hiring manager requested assessments under {duration_constraint} minutes."
        
        # Generate response
        chain = prompt_template | gemini | StrOutputParser()
        response = chain.invoke({
            "query": query,
            "duration_text": duration_text,
            "recommendations": rec_text
        })
        
        return response
    except Exception as e:
        logger.error(f"Error generating natural language response: {e}")
        logger.error(traceback.format_exc())
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
            
            # Extract duration constraint from query
            duration_constraint = extract_duration_constraint(query)
            
            # Search embeddings
            results = search_embeddings(query, url, max_results, duration_constraint)
            processing_time = time.time() - start_time
        
        if results:
            st.success(f"Found {len(results['recommendations'])} recommendations in {processing_time:.2f} seconds using {results.get('search_method', 'unknown')} search!")
            
            # Display enhanced query used
            with st.expander("See how we interpreted your request"):
                st.write(results.get('enhanced_query', 'No enhanced query available'))
            
            # Display recommendations
            st.subheader("Recommended Assessments")
            
            # Display natural language response if available
            if results.get('natural_language_response'):
                st.markdown("### Gemini's Analysis")
                st.markdown(f'<div style="background-color: #404452; padding: 15px; border-radius: 10px; margin-bottom: 20px; color: #D1D5DB;">{results["natural_language_response"]}</div>', unsafe_allow_html=True)
            
            for rec in results['recommendations']:
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <div class="recommendation-title">{rec.get('name', 'Unknown')}</div>
                        <div class="recommendation-detail">
                            <strong>Test Type:</strong> {rec.get('test_type', 'Unknown')} | 
                            <strong>Duration:</strong> {rec.get('duration', 'Unknown')} | 
                            <span class="{'yes-badge' if rec.get('remote_testing') == 'Yes' else 'no-badge'}">Remote Testing: {rec.get('remote_testing', 'No')}</span> | 
                            <span class="{'yes-badge' if rec.get('adaptive_irt') == 'Yes' else 'no-badge'}">Adaptive: {rec.get('adaptive_irt', 'No')}</span> | 
                            <span class="score-badge">Relevance: {rec.get('score', 0)}</span>
                        </div>
                        <div class="recommendation-detail">
                            <a href="{rec.get('url', '#')}" target="_blank">View Assessment Details</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show as a dataframe for easy comparison
            with st.expander("View as table"):
                df = pd.DataFrame(results['recommendations'])
                columns_to_show = ['name', 'test_type', 'duration', 'remote_testing', 'adaptive_irt', 'score']
                columns_to_show = [col for col in columns_to_show if col in df.columns]
                df = df[columns_to_show]
                column_names = {
                    'name': 'Assessment', 
                    'test_type': 'Test Type', 
                    'duration': 'Duration', 
                    'remote_testing': 'Remote Testing', 
                    'adaptive_irt': 'Adaptive/IRT', 
                    'score': 'Relevance Score'
                }
                df.columns = [column_names.get(col, col) for col in df.columns]
                st.dataframe(df)

if __name__ == "__main__":
    main()