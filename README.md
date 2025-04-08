# SHL Assessment Recommendation System

This system helps HR professionals and hiring managers find the most relevant SHL assessments for their job requirements by using advanced natural language processing and vector similarity search.

## Screenshots

### Frontend
![Screenshot 2025-04-08 190704](https://github.com/user-attachments/assets/f776a85f-f375-4579-a446-342206705b06)
### Analysis
![Screenshot 2025-04-08 190744](https://github.com/user-attachments/assets/a4337b72-98a2-41b3-86f2-bd54c6c8e472)
### Analysis Table
![Screenshot 2025-04-08 190754](https://github.com/user-attachments/assets/387930cc-7eb9-4ccb-8acb-2049514837be)

## Features


- **Natural Language Query Processing**: Interprets job descriptions and requirements in natural language
- **Google Gemini-powered Embeddings**: Uses Gemini embeddings for semantic search capabilities
- **MongoDB Integration**: Stores assessment data and embeddings for efficient retrieval
- **Gemini Pro Integration**: Provides natural language explanations of recommendations
- **Duration Filtering**: Automatically filters assessments by duration constraints mentioned in the query
- **URL Integration**: Can extract and process job descriptions from URLs
- **Comprehensive Assessment Matching**: Considers job levels, languages, and test types for better recommendations

## Architecture

The system now operates with a streamlined, Google Gemini-powered architecture:

1. **Data Collection**: Assessments are collected via web scraping and stored in MongoDB
2. **Gemini Embeddings**: Embeddings are created using Google's Gemini embedding model.
3. **Streamlit Frontend**: Connects directly to MongoDB for data and embeddings
4. **Vector Search**: FAISS is used for efficient similarity search
5. **Gemini Pro Integration**: Generates natural language responses about recommendations

## Setup Instructions

### Prerequisites

- Python 3.8+
- MongoDB instance (can be local or cloud-based)
- Google API key for Gemini Pro and Gemini embeddings

### Environment Variables

Create a `.env` file with the following variables:

```
# MongoDB Connection
MONGO_URI=mongodb://username:password@localhost:27017
MONGO_DB=SHL
MONGO_COLLECTION_PACKAGED=packaged_solutions
MONGO_COLLECTION_INDIVIDUAL=individual_solutions
MONGO_EMBEDDINGS_COLLECTION=embeddings

# Google API (for Gemini)
GOOGLE_API_KEY=your_google_api_key
```

### Installation

1. Clone the repository
   ```
   git clone https://github.com/Kaos599/shl-assessment-recommendation.git
   cd shl-assessment-recommendation
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app
   ```
   streamlit run streamlit_app.py
   ```

## Data Collection and Embedding Creation

These steps are not required for normal operation, only for initial setup or updates:

1. Collect assessment data:
   ```
   python data_collection.py
   ```

2. Create embeddings using Gemini:
   ```
   python create_embeddings.py
   ```

   The embedding creation process now includes:
   - Job levels for each assessment
   - Available languages for each assessment
   - Test types and categories
   - Duration information
   - Remote testing capabilities

## Usage

1. Open the Streamlit app in your browser (typically at http://localhost:8501)
2. Enter a job description or requirements in the text field
3. Optionally provide a URL to a job posting
4. Click "Get Recommendations"
5. View the recommendations and explanations

### Note on Hosted Version

The application is hosted on Render.com. Due to Render's free tier policy, the service may take 50 seconds to 2 minutes to respond to the first request after a period of inactivity. This is because the free instance spins down during inactivity and needs time to spin back up. Subsequent requests will be faster until the next period of inactivity.

## Example Queries

- "I need an assessment for Java developers with strong team collaboration skills"
- "Looking for a coding assessment for Python and SQL that takes less than 60 minutes"
- "Need a cognitive assessment for a data analyst position that tests numerical reasoning"
- "Find me an assessment for a senior manager position that's available in Spanish"

## Project Structure

- `streamlit_app.py`: The main Streamlit application
- `create_embeddings.py`: Script for creating and storing embeddings using Google Gemini
- `data_collection.py`: Scripts for collecting assessment data
- `recommendation_engine.py`: Core recommendation logic and query processing
- `embedding_storage.py`: Handles embedding storage and retrieval

## My Implementation Details

In implementing this system, I focused on several key aspects:

1. **Data Quality**: I ensured that the assessment data includes comprehensive information about job levels, languages, and test types to provide more accurate recommendations.

2. **Embedding Creation**: I enhanced the embedding creation process to include job levels and languages, which helps in providing more relevant recommendations based on position seniority and language requirements.

3. **Performance Optimization**: I implemented MongoDB and FAISS for efficient data storage and retrieval, while also handling the cold start issue with Render hosting by clearly communicating the expected response times.

4. **User Experience**: I added natural language processing capabilities using Gemini Pro to provide clear explanations of why certain assessments are recommended, making it easier for HR professionals to make informed decisions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- SHL for their assessment catalog
- Sentence Transformers for the embedding models
- LangChain and Google Gemini for LLM capabilities
