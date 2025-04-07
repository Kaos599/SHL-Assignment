# SHL Assessment Recommendation System - Design Document

## System Architecture

The SHL Assessment Recommendation System is designed as a multi-component application that uses modern AI techniques to match job requirements with appropriate assessment tools from SHL's catalog.

### Architecture Components

1. **Data Collection**
   - Web scraping of SHL product catalog using BeautifulSoup
   - Data cleaning and structuring
   - MongoDB storage with two collections: packaged and individual solutions

2. **Vector Embedding Layer**
   - FAISS vector database for efficient similarity search
   - Dimension: 1536 for optimal performance

3. **Query Processing**
   - Constraint extraction (duration, skills, job level)
   - URL content extraction when provided

4. **Recommendation Engine**
   - Vector similarity search with cosine similarity
   - Constraint filtering
   - Result ranking and formatting

5. **API Layer**
   - FastAPI for RESTful endpoints
   - JSON response format
   - Input validation and error handling

6. **User Interface**
   - Streamlit for interactive web interface
   - Responsive design for multiple devices
   - User-friendly results display

## Key Technical Choices

### Vector Search Implementation

The system uses FAISS for vector similarity search because:
1. **Speed**: Extremely fast retrieval even with large datasets
2. **Accuracy**: High-quality similarity matching
3. **Memory Efficiency**: Optimized for large embedding collections
4. **Integration**: Easy integration with Python ecosystem

### MongoDB Integration

The database design uses two collections:
- `packaged_solutions`: Pre-packaged assessment bundles
- `individual_solutions`: Individual assessments

Each document contains:
- Assessment details (name, URL, duration, etc.)
- Test type information
- Job level suitability
- Remote testing capabilities
- Adaptive/IRT support

## Data Flow

1. **User Input**: Query text or URL
2. **Constraint Extraction**: Identify time/skill constraints
3. **Embedding Generation**: Convert query to vector representation
4. **Vector Search**: Find similar assessments in FAISS index
5. **Filtering**: Apply constraints to results
6. **Ranking**: Order by relevance score
7. **Response**: Return formatted results to user

## Performance Considerations

The system is evaluated using:
- **Mean Recall@K**: Measures ability to retrieve relevant assessments
- **MAP@K**: Evaluates ranking quality
- **Response Time**: < 3 seconds for typical queries
- **Embedding Creation**: Batch processing for efficiency
- **Caching**: Frequently accessed embeddings are cached

## Fallback Mechanisms

The system includes multiple fallback strategies:
1. MongoDB connection issues → Local JSON data
2. FAISS index corruption → Automatic regeneration

## Future Enhancements

1. **Enhanced Multimodal Support**: Process images of job descriptions
2. **Custom Assessment Packages**: Generate custom assessment combinations
3. **User Feedback Loop**: Improve recommendations based on user selections
4. **Expanded Metadata**: Include more detailed assessment characteristics
5. **Cross-lingual Capabilities**: Improved support for non-English queries

## Technology Stack

- **Backend**: Python, FastAPI
- **Frontend**: Streamlit
- **Database**: MongoDB
- **ML/AI**: FAISS for vector similarity search
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