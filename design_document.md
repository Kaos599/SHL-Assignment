# SHL Assessment Recommendation System - Design Document

## Overview
The SHL Assessment Recommendation System is designed to help hiring managers find the most relevant SHL assessments for their job openings. It takes natural language queries, job descriptions, or URLs as input and returns appropriate SHL assessments matched to the requirements.

## Architecture

### Components
1. **Data Collection Module**
   - Scrapes and extracts assessment data from SHL's catalog
   - Preprocesses data for embedding and storage
   - Stores data in MongoDB for persistence and JSON files as backup

2. **Vector Embedding Engine**
   - Converts assessment descriptions into vector embeddings using sentence-transformers
   - Creates a FAISS index for efficient similarity search
   - Enables semantic matching between queries and assessments

3. **Query Enhancement with LLM**
   - Uses Google's Gemini Pro model through LangChain
   - Enhances user queries with relevant skills and competencies
   - Improves semantic matching accuracy

4. **Constraint Extraction**
   - Extracts time/duration constraints from user queries
   - Filters recommendations based on these constraints
   - Ensures results match specific user requirements

5. **FastAPI Backend**
   - Provides REST API endpoints for recommendations
   - Accepts both GET and POST requests
   - Returns structured JSON responses with assessment details

6. **Streamlit Frontend**
   - Provides an intuitive user interface
   - Displays results in a clear, organized format
   - Includes helpful examples and explanations

## Data Flow

1. **Data Collection and Preparation**
   - Assessment data is scraped from SHL catalog
   - Data is cleaned and preprocessed
   - Text features are combined for embedding
   - Data is stored in MongoDB and as JSON files

2. **Vector Embedding Creation**
   - Assessment descriptions are converted to vector embeddings
   - A FAISS index is created for similarity search
   - Model and index are saved for reuse

3. **Query Processing**
   - User submits a query, job description, or URL
   - If URL is provided, content is extracted and added to the query
   - Query is enhanced using Gemini Pro LLM
   - Time constraints are extracted if present

4. **Recommendation Generation**
   - Enhanced query is converted to a vector embedding
   - Similar assessments are found using FAISS
   - Results are filtered based on constraints
   - Top recommendations are returned with details

## Technology Stack

- **Backend**: Python, FastAPI
- **Frontend**: Streamlit
- **Database**: MongoDB
- **ML/AI**: 
  - Sentence Transformers for embeddings
  - FAISS for vector similarity search
  - Google Gemini Pro via LangChain for query enhancement
- **Data Processing**: Pandas, NumPy
- **Web Scraping**: BeautifulSoup, Requests

## Evaluation

The system is evaluated using the following metrics:
- **Mean Recall@K**: Measures ability to find relevant assessments
- **Mean Average Precision@K**: Measures ranking quality of results
- **Processing Time**: Measures efficiency of recommendation generation

Test queries include:
1. "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
2. "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes."
3. "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins."

## Future Improvements

1. **Expanded Data Collection**: Add more assessment details and descriptions
2. **Improved Query Understanding**: Enhanced context analysis for better matching
3. **User Feedback Loop**: Incorporate user feedback to improve recommendations
4. **Multi-modal Input**: Support for image and document uploads (e.g., job descriptions as PDFs)
5. **Assessment Packages**: Generate customized assessment combinations for specific roles 