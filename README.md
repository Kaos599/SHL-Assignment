# SHL Assessment Recommendation System

This system helps HR professionals and hiring managers find the most relevant SHL assessments for their job requirements by using advanced natural language processing and vector similarity search.

## Features

- **Natural Language Query Processing**: Interprets job descriptions and requirements in natural language
- **Google Gemini-powered Embeddings**: Uses Gemini embeddings for semantic search capabilities
- **MongoDB Integration**: Stores assessment data and embeddings for efficient retrieval
- **Gemini Pro Integration**: Provides natural language explanations of recommendations
- **Duration Filtering**: Automatically filters assessments by duration constraints mentioned in the query
- **URL Integration**: Can extract and process job descriptions from URLs

## Architecture

The system now operates with a streamlined, Google Gemini-powered architecture:

1. **Data Collection**: Assessments are collected via web scraping and stored in MongoDB
2. **Gemini Embeddings**: Embeddings are created using Google's Gemini embedding model
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
   git clone https://github.com/yourusername/shl-assessment-recommendation.git
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

## Usage

1. Open the Streamlit app in your browser (typically at http://localhost:8501)
2. Enter a job description or requirements in the text field
3. Optionally provide a URL to a job posting
4. Click "Get Recommendations"
5. View the recommendations and explanations

## Example Queries

- "I need an assessment for Java developers with strong team collaboration skills"
- "Looking for a coding assessment for Python and SQL that takes less than 60 minutes"
- "Need a cognitive assessment for a data analyst position that tests numerical reasoning"

## Project Structure

- `streamlit_app.py`: The main Streamlit application
- `create_embeddings.py`: Script for creating and storing embeddings using Google Gemini
- `data_collection.py`: Scripts for collecting assessment data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- SHL for their assessment catalog
- Sentence Transformers for the embedding models
- LangChain and Google Gemini for LLM capabilities