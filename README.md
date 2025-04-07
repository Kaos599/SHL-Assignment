# SHL Assessment Recommendation System

## Project Overview
An intelligent recommendation system that helps hiring managers find the most relevant SHL assessments for their job openings. The system takes natural language queries or job descriptions as input and returns appropriate SHL assessments from the SHL product catalog.

## Features

- **Natural Language Input**: Processes queries about job requirements and needed assessment types
- **Multi-source Input**: Accepts text queries, job descriptions, or URLs to job postings
- **Smart Filtering**: Filters results based on constraints like time limits
- **Comprehensive Results**: Returns up to 10 relevant assessments with complete details

## Technical Implementation

### System Architecture

The system is built with a hybrid architecture combining:

1. **Data Collection Pipeline**: Web scraping for SHL product data
2. **Vector Database**: FAISS for semantic search
3. **API Backend**: FastAPI
4. **Web Interface**: Streamlit frontend

### Components

- **Data Collection (`data_collection.py`)**: Scrapes SHL product catalog 
- **Embedding Generation (`embedding.py`)**: Creates embeddings
- **Recommendation Engine (`recommendation_engine.py`)**: Core matching logic
- **API (`app.py`)**: REST API endpoints
- **Web UI (`streamlit_app.py`)**: User interface
- **Evaluation Framework (`evaluate.py`)**: Test framework for measuring performance

## Vector Search Benefits

- **Semantic Understanding**: Captures meaning beyond keyword matching
- **Multilingual Support**: Works across languages through semantic understanding
- **Relevance Ranking**: Orders results by semantic similarity score
- **Efficient Retrieval**: Fast search even with large assessment databases

## Running the System

### Prerequisites

- Python 3.9+
- MongoDB for data storage

### Environment Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`:
```
# MongoDB Connection
MONGO_URI="your-mongodb-connection-string"
MONGO_DB="SHL"
MONGO_COLLECTION_PACKAGED="packaged_solutions"
MONGO_COLLECTION_INDIVIDUAL="individual_solutions"
```

### Running the System

1. Generate embeddings:
```bash
python embedding.py
```

2. Start the API:
```bash
python app.py
```

3. Launch the web interface:
```bash
streamlit run streamlit_app.py
```

## Evaluation

The system performance can be evaluated using:

```bash
python evaluate.py
```

This calculates:
- Mean Recall@K (K=1,3,5,10)
- Mean Average Precision@K (MAP@K)

## Example Queries

Try these sample queries:

1. "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."

2. "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript. Need an assessment package that can test all skills with max duration of 60 minutes."

3. "I am hiring for an analyst and want to screen applications using Cognitive and personality tests. What options are available within 45 minutes?"

## Project Structure

```
shl-recommender/
├── app.py                      # FastAPI backend
├── data_collection.py          # Web scraping
├── embedding.py                # Vector embeddings
├── evaluate.py                 # Evaluation metrics
├── recommendation_engine.py    # Core engine
├── streamlit_app.py            # Web interface
├── requirements.txt            # Dependencies
├── design_document.md          # Approach explanation
├── README.md                   # Instructions
├── .env                        # Configuration
└── data/                       # Generated data
    ├── shl_assessments.json
    ├── embedding_model.pkl
    ├── assessment_index.faiss
```

## License

MIT

## Acknowledgements

- SHL for their assessment catalog
- Sentence Transformers for the embedding models
- LangChain and Google Gemini for LLM capabilities