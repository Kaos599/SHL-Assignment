import os
import faiss
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB connection settings
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "SHL")
MONGO_EMBEDDINGS_COLLECTION = os.getenv("MONGO_EMBEDDINGS_COLLECTION", "embeddings")

def get_mongo_client():
    """Get MongoDB client connection."""
    return MongoClient(MONGO_URI)

def create_faiss_index():
    """Create FAISS index from embeddings stored in MongoDB."""
    try:
        # Connect to MongoDB
        client = get_mongo_client()
        db = client[MONGO_DB]
        embeddings_collection = db[MONGO_EMBEDDINGS_COLLECTION]
        
        # Load embeddings from MongoDB
        embeddings = []
        assessment_ids = []
        
        for doc in embeddings_collection.find():
            embeddings.append(doc['embedding'])
            assessment_ids.append(doc['assessment_id'])
        
        if not embeddings:
            logger.error("No embeddings found in MongoDB")
            return
        
        # Convert to numpy array
        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]
        
        logger.info(f"Loaded {len(embeddings)} embeddings of dimension {dimension}")
        
        # Create FAISS index
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        # Save index
        os.makedirs('data', exist_ok=True)
        faiss.write_index(index, 'data/assessment_index.faiss')
        
        # Save assessment IDs for reference
        with open('data/assessment_ids.txt', 'w') as f:
            for id in assessment_ids:
                f.write(f"{id}\n")
        
        logger.info(f"Created and saved FAISS index with {index.ntotal} vectors")
        
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    create_faiss_index() 