# SHL Assessment Recommendation System

An intelligent recommendation system that helps hiring managers find the most relevant SHL assessments for their job openings. The system takes natural language queries or job descriptions as input and returns appropriate SHL assessments from the SHL product catalog.

## Features

- Take input as natural language query, job description text, or URL to a job posting
- Return up to 10 relevant SHL assessments with detailed information
- Implementation of filtering based on constraints (e.g., "completed in 40 minutes")
- LLM-powered query enhancement using Google Gemini Pro
- MongoDB integration for data storage
- Vector embeddings for semantic search
- FastAPI backend with REST API endpoints
- Streamlit frontend for an intuitive user interface

## System Architecture

![System Architecture](https://i.ibb.co/tX00Vrm/shl-architecture.png)

The system consists of several components:
1. Data Collection Module
2. Vector Embedding Engine
3. Query Enhancement with LLM
4. Constraint Extraction
5. FastAPI Backend
6. Streamlit Frontend

For more details, see the [Design Document](design_document.md).

## Installation

### Prerequisites

- Python 3.9+
- MongoDB
- Google API Key (for Gemini Pro)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/shl-assessment-recommender.git
   cd shl-assessment-recommender
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your credentials:
   ```
   GOOGLE_API_KEY="your-google-api-key"
   MONGO_URI="your-mongodb-connection-string"
   MONGO_DB="SHL"
   MONGO_COLLECTION="data"
   ```

## Usage

### Data Collection

Run the data collection script to scrape and store SHL assessment data:

```bash
python data_collection.py
```

### Create Embeddings

Generate vector embeddings for the assessments:

```bash
python embedding.py
```

### Run the API

Start the FastAPI backend:

```bash
uvicorn app:app --reload
```

The API will be available at http://localhost:8000.

### Run the Web Interface

Start the Streamlit frontend:

```bash
streamlit run streamlit_app.py
```

The web interface will be available at http://localhost:8501.

## API Endpoints

- `GET /`: API status check
- `GET /recommend`: Get assessment recommendations via GET
- `POST /recommend`: Get assessment recommendations via POST

Example API call:
```bash
curl -X GET "http://localhost:8000/recommend?query=Java%20developers%20who%20can%20collaborate%20with%20business%20teams&max_results=5"
```

## Evaluation

The system is evaluated using Mean Recall@K and MAP@K metrics. Run the evaluation script:

```bash
python evaluate.py
```

## Project Structure

```
shl-recommender/
├── app.py                      # FastAPI backend
├── data_collection.py          # Web scraping
├── embedding.py                # Vector embeddings
├── evaluate.py                 # Evaluation metrics
├── recommendation_engine.py    # Core engine with LLM
├── streamlit_app.py            # Web interface
├── requirements.txt            # Dependencies
├── design_document.md          # Approach explanation
├── README.md                   # Instructions
├── .env                        # API keys (create locally)
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