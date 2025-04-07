import os
import json
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import logging
from tqdm import tqdm
import time
import traceback
import numpy as np
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from embedding_storage import EmbeddingStorage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "SHL")
MONGO_COLLECTION_PACKAGED = os.getenv("MONGO_COLLECTION_PACKAGED", "packaged_solutions")
MONGO_COLLECTION_INDIVIDUAL = os.getenv("MONGO_COLLECTION_INDIVIDUAL", "individual_solutions")
MONGO_EMBEDDINGS_COLLECTION = os.getenv("MONGO_EMBEDDINGS_COLLECTION", "embeddings")

# Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

class GeminiEmbeddingStorage:
    """Class for creating and storing embeddings using Google's Gemini API."""
    
    def __init__(self):
        """Initialize embedding storage with MongoDB connection."""
        self.client = None
        self.embedding_model = None
        self.embedding_dimension = 768  # Default for Gemini embeddings
        
        # Initialize MongoDB client
        try:
            self.client = MongoClient(
                MONGO_URI,
                connectTimeoutMS=60000,
                socketTimeoutMS=90000,
                serverSelectionTimeoutMS=60000,
                maxPoolSize=100,
                retryWrites=True,
                retryReads=True
            )
            # Test the connection
            self.client.admin.command('ping')
            self.db = self.client[MONGO_DB]
            self.embeddings_collection = self.db[MONGO_EMBEDDINGS_COLLECTION]
            logger.info(f"Connected to MongoDB collection: {MONGO_EMBEDDINGS_COLLECTION}")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            traceback.print_exc()
            raise
        
        # Initialize Gemini embedding model
        try:
            # LangChain wrapper provides a simpler interface
            self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            # Test the model with a simple embedding request
            test_embedding = self.get_embedding("This is a test")
            self.embedding_dimension = len(test_embedding)
            logger.info(f"Successfully initialized Gemini embedding model. Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Error initializing Gemini embedding model: {e}")
            traceback.print_exc()
            raise
            
    def get_embedding(self, text):
        """Get embedding from Gemini API."""
        if not self.embedding_model:
            logger.error("Gemini embedding model not initialized")
            return None
            
        try:
            # Using LangChain wrapper for consistent interface
            embedding = self.embedding_model.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error getting Gemini embedding: {e}")
            traceback.print_exc()
            
            # Try with direct API as fallback
            try:
                logger.info("Trying direct Gemini API as fallback")
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document")
                embedding = result["embedding"]
                return embedding
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return None
            
    def store_embedding(self, assessment_id, text, embedding):
        """Store an embedding in MongoDB."""
        if not self.client:
            logger.error("MongoDB client not initialized")
            return False
            
        try:
            # Check if an embedding already exists for this assessment
            existing = self.embeddings_collection.find_one({"assessment_id": assessment_id})
            
            if existing:
                # Update existing embedding
                self.embeddings_collection.update_one(
                    {"assessment_id": assessment_id},
                    {"$set": {
                        "text": text,
                        "embedding": embedding,
                        "embedding_model": "gemini-embedding-001"
                    }}
                )
                logger.info(f"Updated embedding for assessment ID: {assessment_id}")
            else:
                # Insert new embedding
                self.embeddings_collection.insert_one({
                    "assessment_id": assessment_id,
                    "text": text,
                    "embedding": embedding,
                    "embedding_model": "gemini-embedding-001"
                })
                logger.info(f"Stored new embedding for assessment ID: {assessment_id}")
                
            return True
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            traceback.print_exc()
            return False
            
    def get_stored_embedding(self, assessment_id):
        """Retrieve a stored embedding from MongoDB."""
        if not self.client:
            logger.error("MongoDB client not initialized")
            return None
            
        try:
            result = self.embeddings_collection.find_one({"assessment_id": assessment_id})
            if result and "embedding" in result:
                return result["embedding"]
            return None
        except Exception as e:
            logger.error(f"Error retrieving embedding: {e}")
            return None
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

def get_mongodb_connection():
    """Create MongoDB connection with extended timeouts."""
    try:
        client = MongoClient(
            MONGO_URI,
            connectTimeoutMS=60000,
            socketTimeoutMS=90000,
            serverSelectionTimeoutMS=60000,
            maxPoolSize=100,
            retryWrites=True,
            retryReads=True
        )
        # Test connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        traceback.print_exc()
        return None

def load_data_from_collection(client, collection_name, limit=None, chunk_size=50):
    """Load data from MongoDB collection with chunking to avoid timeouts."""
    try:
        db = client[MONGO_DB]
        collection = db[collection_name]
        
        # Count documents first
        count = collection.count_documents({})
        logger.info(f"Found {count} documents in {collection_name}")
        
        # Initialize empty dataframe to hold results
        results_df = pd.DataFrame()
        
        # Apply limit if specified
        total_docs = limit if limit and limit < count else count
        
        # Process in chunks
        for skip in range(0, total_docs, chunk_size):
            # Calculate actual chunk size for this iteration
            current_chunk_size = min(chunk_size, total_docs - skip)
            logger.info(f"Processing chunk of {current_chunk_size} documents (skip={skip})")
            
            # Get chunk of documents
            try:
                chunk_cursor = collection.find({}).skip(skip).limit(current_chunk_size)
                chunk_data = list(chunk_cursor)
                
                if not chunk_data:
                    logger.warning(f"No data found in chunk (skip={skip}, limit={current_chunk_size})")
                    continue
                    
                # Convert to dataframe
                chunk_df = pd.DataFrame(chunk_data)
                
                # Create text_for_embedding
                chunk_df['text_for_embedding'] = chunk_df.apply(
                    lambda row: f"name: {row.get('name', 'Unknown')}\n"
                               f"adaptive_irt: {row.get('adaptive_irt', False)}\n"
                               f"remote_testing: {row.get('remote_testing', False)}\n"
                               f"test_type: {', '.join(str(t) for t in row.get('test_type', []))}\n" 
                               f"test_type_names: {', '.join(str(t) for t in row.get('test_type_names', []))}\n"
                               f"url: {row.get('url', '')}\n"
                               f"category: {row.get('category', '')}\n"
                               f"duration_minutes: {row.get('duration_minutes', 0)}",
                    axis=1
                )
                
                # Append to results
                results_df = pd.concat([results_df, chunk_df], ignore_index=True)
                logger.info(f"Processed {len(results_df)}/{total_docs} documents from {collection_name}")
                
                # Sleep to avoid overwhelming the MongoDB server
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error processing chunk (skip={skip}, limit={current_chunk_size}): {e}")
                traceback.print_exc()
                # Continue with next chunk
                time.sleep(5)  # Wait longer after an error
                continue
        
        if results_df.empty:
            logger.warning(f"No data found in {collection_name}")
            return pd.DataFrame()
            
        logger.info(f"Successfully loaded {len(results_df)} documents from {collection_name}")
        return results_df
        
    except Exception as e:
        logger.error(f"Error loading data from {collection_name}: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def process_single_item(embedding_storage, id_value, text):
    """Process a single item and create/store its embedding."""
    try:
        # Check if embedding already exists
        existing_embedding = embedding_storage.get_stored_embedding(id_value)
        if existing_embedding:
            logger.info(f"Embedding already exists for ID: {id_value}")
            return True
            
        # Generate embedding
        embedding = embedding_storage.get_embedding(text)
        if not embedding:
            logger.error(f"Failed to generate embedding for ID: {id_value}")
            return False
            
        # Store embedding
        success = embedding_storage.store_embedding(id_value, text, embedding)
        if success:
            logger.info(f"Successfully stored embedding for ID: {id_value}")
            return True
        else:
            logger.error(f"Failed to store embedding for ID: {id_value}")
            return False
    except Exception as e:
        logger.error(f"Error processing item {id_value}: {e}")
        traceback.print_exc()
        return False

def main():
    """Main function to create and store embeddings using Gemini."""
    # Connect to MongoDB
    client = get_mongodb_connection()
    if not client:
        logger.error("Failed to connect to MongoDB. Exiting.")
        return
        
    # Initialize Gemini embedding storage
    embedding_storage = GeminiEmbeddingStorage()
    
    # Load data from collections
    # Process all documents by not specifying a limit
    test_limit = None  # Process all documents
    
    # Define chunk sizes for different collections
    packaged_chunk_size = 50
    individual_chunk_size = 30  # Smaller chunk size for individual solutions
    
    logger.info("Loading packaged solutions...")
    packaged_df = load_data_from_collection(client, MONGO_COLLECTION_PACKAGED, limit=test_limit, chunk_size=packaged_chunk_size)
    
    logger.info("Loading individual solutions...")
    individual_df = load_data_from_collection(client, MONGO_COLLECTION_INDIVIDUAL, limit=test_limit, chunk_size=individual_chunk_size)
    
    # Process packaged solutions in chunks
    if not packaged_df.empty:
        logger.info(f"Processing {len(packaged_df)} packaged solutions...")
        # Process in chunks of 50
        for chunk_start in range(0, len(packaged_df), 50):
            chunk_end = min(chunk_start + 50, len(packaged_df))
            chunk = packaged_df.iloc[chunk_start:chunk_end]
            logger.info(f"Processing packaged solutions chunk {chunk_start}-{chunk_end}")
            
            for idx, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"Processing packaged chunk {chunk_start}-{chunk_end}"):
                id_value = str(row.get('_id', f"packaged_{idx}"))
                text = row['text_for_embedding']
                process_single_item(embedding_storage, id_value, text)
    
    # Process individual solutions in chunks
    if not individual_df.empty:
        logger.info(f"Processing {len(individual_df)} individual solutions...")
        # Process in chunks of 30
        for chunk_start in range(0, len(individual_df), 30):
            chunk_end = min(chunk_start + 30, len(individual_df))
            chunk = individual_df.iloc[chunk_start:chunk_end]
            logger.info(f"Processing individual solutions chunk {chunk_start}-{chunk_end}")
            
            for idx, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"Processing individual chunk {chunk_start}-{chunk_end}"):
                id_value = str(row.get('_id', f"individual_{idx}"))
                text = row['text_for_embedding']
                process_single_item(embedding_storage, id_value, text)
            
            # Sleep between chunks to avoid overwhelming the server
            logger.info("Sleeping between chunks...")
            time.sleep(5)
    
    # Verify embeddings in MongoDB
    try:
        db = client[MONGO_DB]
        embeddings_collection = db[MONGO_EMBEDDINGS_COLLECTION]
        count = embeddings_collection.count_documents({})
        logger.info(f"Total embeddings in MongoDB: {count}")
    except Exception as e:
        logger.error(f"Error counting embeddings: {e}")
    
    # Close connections
    embedding_storage.close()
    client.close()
    
    logger.info("Gemini embedding creation process completed")

if __name__ == "__main__":
    main() 