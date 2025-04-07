import pandas as pd
import numpy as np
import json
import os
from openai import AzureOpenAI
import faiss
import pickle
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm
import sys
import traceback

# Load environment variables
load_dotenv()

# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "SHL")
MONGO_COLLECTION_PACKAGED = os.getenv("MONGO_COLLECTION_PACKAGED", "packaged_solutions")
MONGO_COLLECTION_INDIVIDUAL = os.getenv("MONGO_COLLECTION_INDIVIDUAL", "individual_solutions")
MONGO_CONNECT_TIMEOUT_MS = int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "30000"))
MONGO_SOCKET_TIMEOUT_MS = int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "45000"))

# Azure OpenAI connection
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")

# Validate Azure OpenAI environment variables
if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_VERSION or not AZURE_EMBEDDING_DEPLOYMENT_NAME:
    logger.error("Missing Azure OpenAI environment variables!")
    logger.error(f"API Key: {'Set' if AZURE_OPENAI_API_KEY else 'Missing'}")
    logger.error(f"Endpoint: {AZURE_OPENAI_ENDPOINT or 'Missing'}")
    logger.error(f"API Version: {AZURE_OPENAI_API_VERSION or 'Missing'}")
    logger.error(f"Embedding Model: {AZURE_EMBEDDING_DEPLOYMENT_NAME or 'Missing'}")

# Print environment variables (with masked key)
logger.info(f"MongoDB URI: {MONGO_URI[:10]}...{MONGO_URI[-5:] if MONGO_URI else 'None'}")
logger.info(f"MongoDB DB: {MONGO_DB}")
logger.info(f"MongoDB Collections: {MONGO_COLLECTION_PACKAGED}, {MONGO_COLLECTION_INDIVIDUAL}")
logger.info(f"MongoDB Connect Timeout: {MONGO_CONNECT_TIMEOUT_MS}ms, Socket Timeout: {MONGO_SOCKET_TIMEOUT_MS}ms")
logger.info(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT}")
logger.info(f"Azure OpenAI API Version: {AZURE_OPENAI_API_VERSION}")
logger.info(f"Azure Embedding Model: {AZURE_EMBEDDING_DEPLOYMENT_NAME}")

def get_mongo_client():
    """Get MongoDB client connection."""
    logger.info("Creating MongoDB client connection")
    client = MongoClient(
        MONGO_URI,
        connectTimeoutMS=MONGO_CONNECT_TIMEOUT_MS,
        socketTimeoutMS=MONGO_SOCKET_TIMEOUT_MS,
        serverSelectionTimeoutMS=MONGO_CONNECT_TIMEOUT_MS
    )
    return client

def load_data_from_mongodb():
    """Load assessment data from MongoDB."""
    packaged_data = []
    individual_data = []
    
    try:
        logger.info("Loading data from MongoDB...")
        client = get_mongo_client()
        db = client[MONGO_DB]
        
        # Load packaged solutions
        try:
            logger.info(f"Loading data from {MONGO_COLLECTION_PACKAGED} collection")
            packaged_collection = db[MONGO_COLLECTION_PACKAGED]
            packaged_cursor = packaged_collection.find({})
            packaged_data = list(packaged_cursor)
            logger.info(f"Loaded {len(packaged_data)} packaged solutions")
        except Exception as e:
            logger.error(f"Error loading packaged solutions: {e}")
        
        # Load individual solutions
        try:
            logger.info(f"Loading data from {MONGO_COLLECTION_INDIVIDUAL} collection")
            individual_collection = db[MONGO_COLLECTION_INDIVIDUAL]
            individual_cursor = individual_collection.find({})
            individual_data = list(individual_cursor)
            logger.info(f"Loaded {len(individual_data)} individual solutions")
        except Exception as e:
            logger.error(f"Error loading individual solutions: {e}")
        
        # Combine data
        data = packaged_data + individual_data
        
        if not data:
            logger.warning("No data found in MongoDB collections")
            return None
        
        # Convert MongoDB data to DataFrame
        df = pd.DataFrame(data)
        
        logger.info(f"Successfully loaded {len(df)} assessments from MongoDB")
        # Print column names
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Error loading from MongoDB: {e}")
        traceback.print_exc()
        return None
    finally:
        client.close()

def load_data_from_json(file_path='data/shl_assessments.json'):
    """Load assessment data from a JSON file as a fallback."""
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if os.path.exists(file_path):
            assessments_df = pd.read_json(file_path)
            logger.info(f"Loaded {len(assessments_df)} assessments from JSON file: {file_path}")
            return assessments_df
        else:
            logger.error(f"JSON file not found: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading JSON data: {e}")
        traceback.print_exc()
        return None

def prepare_text_for_embedding(row):
    """Prepare text for embedding by combining relevant fields."""
    text_fields = []
    
    # Add name
    if 'name' in row and row['name']:
        text_fields.append(f"Name: {row['name']}")
    
    # Add description
    if 'description' in row and row['description']:
        text_fields.append(f"Description: {row['description']}")
    
    # Add test types
    if 'test_type_names' in row and row['test_type_names']:
        test_types = ', '.join(row['test_type_names'])
        text_fields.append(f"Test Types: {test_types}")
    
    # Add job levels if available
    if 'job_levels' in row and row['job_levels']:
        job_levels = ', '.join(row['job_levels'])
        text_fields.append(f"Job Levels: {job_levels}")
    
    # Add duration
    if 'duration_minutes' in row and row['duration_minutes']:
        text_fields.append(f"Duration: {row['duration_minutes']} minutes")
    
    # Add solution type if available
    if 'solution_type' in row and row['solution_type']:
        text_fields.append(f"Solution Type: {row['solution_type']}")
    
    # Add category if available
    if 'category' in row and row['category']:
        text_fields.append(f"Category: {row['category']}")
    
    # Join all fields with spaces
    return ' '.join(text_fields)

def get_azure_openai_embedding(text, client):
    """Get embedding from Azure OpenAI API."""
    try:
        response = client.embeddings.create(
            input=text,
            model=AZURE_EMBEDDING_DEPLOYMENT_NAME
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        traceback.print_exc()
        return None

def create_embeddings(assessments_df):
    """Convert assessment descriptions into vector embeddings using Azure OpenAI."""
    logger.info("Setting up Azure OpenAI client...")
    try:
        # Validate Azure OpenAI environment variables again before creating client
        if not AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY is not set or is empty")
        if not AZURE_OPENAI_ENDPOINT:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set or is empty")
        if not AZURE_OPENAI_API_VERSION:
            raise ValueError("AZURE_OPENAI_API_VERSION is not set or is empty")
        
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        
        logger.info("Preparing text for embeddings...")
        assessments_df['text_for_embedding'] = assessments_df.apply(prepare_text_for_embedding, axis=1)
        
        # Print first few texts for embedding
        logger.info("Sample text for embedding:")
        for i in range(min(3, len(assessments_df))):
            logger.info(f"Sample {i}: {assessments_df['text_for_embedding'].iloc[i][:100]}...")
        
        logger.info("Creating embeddings for assessments...")
        embeddings = []
        
        # Test embedding with a single sample first
        logger.info("Testing embedding with a single sample...")
        sample_text = assessments_df['text_for_embedding'].iloc[0]
        sample_embedding = get_azure_openai_embedding(sample_text, client)
        if sample_embedding:
            logger.info(f"Sample embedding successfully created with length {len(sample_embedding)}")
            logger.info(f"Sample embedding first 5 values: {sample_embedding[:5]}")
        else:
            logger.error("Failed to create sample embedding")
            return None
        
        for idx, row in tqdm(assessments_df.iterrows(), total=len(assessments_df), desc="Processing embeddings"):
            text = row['text_for_embedding']
            embedding = get_azure_openai_embedding(text, client)
            if embedding:
                embeddings.append(embedding)
            else:
                # If embedding fails, add a zero vector of the same dimension as successful embeddings
                # We'll determine the dimension from the first successful embedding
                if embeddings:
                    embeddings.append([0.0] * len(embeddings[0]))
                else:
                    # If this is the first item and it failed, we'll add a placeholder and fix it later
                    embeddings.append([0.0])
        
        assessments_df['embedding'] = embeddings
        
        # Create a dictionary with the Azure OpenAI client info for later use
        azure_info = {
            "api_version": AZURE_OPENAI_API_VERSION,
            "embedding_model": AZURE_EMBEDDING_DEPLOYMENT_NAME,
            "endpoint": AZURE_OPENAI_ENDPOINT
        }
        
        os.makedirs('data', exist_ok=True)
        with open('data/azure_openai_info.pkl', 'wb') as f:
            pickle.dump(azure_info, f)
        
        return assessments_df
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        traceback.print_exc()
        return None

def create_faiss_index(embeddings):
    """Create a FAISS index for fast vector similarity search."""
    try:
        logger.info(f"Creating FAISS index with {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        embedding_matrix = np.array(embeddings).astype('float32')
        dimension = embedding_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embedding_matrix)
        index.add(embedding_matrix)
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        faiss.write_index(index, 'data/assessment_index.faiss')
        logger.info("FAISS index saved to data/assessment_index.faiss")
        
        return index
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        traceback.print_exc()
        return None

def save_processed_data(df, filename='data/processed_assessments.json'):
    """Save the processed data without embeddings to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert ObjectId to string for JSON serialization
        df_copy = df.copy()
        if '_id' in df_copy.columns:
            df_copy['_id'] = df_copy['_id'].astype(str)
        
        # Remove embedding column for JSON storage
        if 'embedding' in df_copy.columns:
            df_copy = df_copy.drop(columns=['embedding'])
        
        # Save as JSON
        json_data = df_copy.to_json(orient='records', indent=4)
        with open(filename, 'w') as f:
            f.write(json_data)
        logger.info(f"Processed data saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        traceback.print_exc()

def create_sample_data():
    """Create sample data for testing when no other data sources are available."""
    logger.info("Creating sample data for testing...")
    
    sample_data = [
        {
            "name": "Leadership Assessment",
            "description": "A comprehensive assessment for leadership potential and skills.",
            "test_type_names": ["Personality & Behavior", "Competencies"],
            "job_levels": ["Mid-level", "Senior"],
            "duration_minutes": 45,
            "remote_testing": True,
            "adaptive_irt": False,
            "solution_type": "packaged",
            "category": "Leadership"
        },
        {
            "name": "Technical Skills - Java",
            "description": "Assessment for measuring Java programming skills and problem-solving ability.",
            "test_type_names": ["Knowledge & Skills", "Ability & Aptitude"],
            "job_levels": ["Entry-Level", "Mid-level"],
            "duration_minutes": 60,
            "remote_testing": True,
            "adaptive_irt": True,
            "solution_type": "individual",
            "category": "Technical Skills"
        },
        {
            "name": "Customer Service Package",
            "description": "Comprehensive package for assessing customer service skills and aptitude.",
            "test_type_names": ["Personality & Behavior", "Competencies", "Situational Judgement"],
            "job_levels": ["Entry-Level"],
            "duration_minutes": 40,
            "remote_testing": True,
            "adaptive_irt": False,
            "solution_type": "packaged",
            "category": "Customer Service"
        }
    ]
    
    df = pd.DataFrame(sample_data)
    logger.info(f"Created sample data with {len(df)} assessments")
    
    # Save sample data
    os.makedirs('data', exist_ok=True)
    with open('data/sample_assessments.json', 'w') as f:
        json.dump(sample_data, f, indent=4)
    
    return df

if __name__ == "__main__":
    logger.info("Starting embedding generation process...")
    
    try:
        # Load data from MongoDB
        assessments_df = load_data_from_mongodb()
        
        # If MongoDB load fails, try to load from JSON
        if assessments_df is None or assessments_df.empty:
            logger.warning("Falling back to JSON data source...")
            assessments_df = load_data_from_json()
            
            # If JSON load fails too, create sample data
            if assessments_df is None or assessments_df.empty:
                logger.warning("No data found in JSON file. Creating sample data...")
                assessments_df = create_sample_data()
        
        # Create embeddings
        assessments_df = create_embeddings(assessments_df)
        
        if assessments_df is None:
            logger.error("Failed to create embeddings")
            sys.exit(1)
        
        # Create FAISS index
        if 'embedding' in assessments_df.columns and not assessments_df['embedding'].isna().any():
            embeddings = np.vstack(assessments_df['embedding'].values)
            index = create_faiss_index(embeddings)
            if index:
                logger.info(f"Created FAISS index with {len(embeddings)} embeddings")
            else:
                logger.error("Failed to create FAISS index")
                sys.exit(1)
        else:
            logger.error("Error: Missing embeddings in dataset")
            sys.exit(1)
        
        # Save processed data
        save_processed_data(assessments_df)
        
        logger.info("Embedding and indexing complete!")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)