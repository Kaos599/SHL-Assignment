import pandas as pd
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "SHL")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "data")

def get_mongo_client():
    """Get MongoDB client connection."""
    client = MongoClient(MONGO_URI)
    return client

def load_data_from_mongodb():
    """Load assessment data from MongoDB."""
    try:
        client = get_mongo_client()
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        cursor = collection.find({})
        data = list(cursor)
        
        if not data:
            print("No data found in MongoDB collection")
            return None
        
        df = pd.DataFrame(data)
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        print(f"Successfully loaded {len(df)} assessments from MongoDB")
        return df
    except Exception as e:
        print(f"Error loading from MongoDB: {e}")
        return None
    finally:
        client.close()

def create_embeddings(assessments_df):
    """Convert assessment descriptions into vector embeddings."""
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Creating embeddings for assessments...")
    texts = assessments_df['text_for_embedding'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    
    assessments_df['embedding'] = list(embeddings)
    
    with open('data/embedding_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return assessments_df, model

def create_faiss_index(embeddings):
    """Create a FAISS index for fast vector similarity search."""
    embedding_matrix = np.array(embeddings).astype('float32')
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embedding_matrix)
    index.add(embedding_matrix)
    
    faiss.write_index(index, 'data/assessment_index.faiss')
    
    return index

def save_processed_data(df, filename='data/processed_assessments.json'):
    """Save the processed data without embeddings to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df_save = df.drop(columns=['embedding'])
    json_data = df_save.to_json(orient='records', indent=4)
    with open(filename, 'w') as f:
        f.write(json_data)
    print(f"Processed data saved to {filename}")

if __name__ == "__main__":
    assessments_df = load_data_from_mongodb()
    
    if assessments_df is None or assessments_df.empty:
        print("Falling back to JSON data source...")
        assessments_df = pd.read_json('data/shl_assessments.json')
    
    assessments_df, model = create_embeddings(assessments_df)
    
    embeddings = np.vstack(assessments_df['embedding'].values)
    index = create_faiss_index(embeddings)
    
    save_processed_data(assessments_df)
    
    print("Embedding and indexing complete!")