import os
import json
import numpy as np
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
import logging
from tqdm import tqdm
import traceback
import time
import google.generativeai as google_genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "SHL")
MONGO_EMBEDDINGS_COLLECTION = os.getenv("MONGO_EMBEDDINGS_COLLECTION", "embeddings")
MONGO_CONNECT_TIMEOUT_MS = int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "60000"))  # Increased to 60 seconds
MONGO_SOCKET_TIMEOUT_MS = int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "90000"))  # Increased to 90 seconds
MONGO_MAX_POOL_SIZE = int(os.getenv("MONGO_MAX_POOL_SIZE", "100"))
MONGO_RETRY_WRITES = os.getenv("MONGO_RETRY_WRITES", "true").lower() == "true"
MONGO_RETRY_READS = os.getenv("MONGO_RETRY_READS", "true").lower() == "true"

# Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class EmbeddingStorage:
    """Class for storing and retrieving embeddings in MongoDB."""
    
    def __init__(self):
        """Initialize embedding storage with MongoDB connection."""
        self.client = None
        self.embedding_dimension = 768  # Default for Gemini embeddings
        
        # Configure Google API
        google_genai.configure(api_key=GOOGLE_API_KEY)
        
        # Connect to MongoDB with retry logic
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                self.client = MongoClient(
                    MONGO_URI,
                    connectTimeoutMS=MONGO_CONNECT_TIMEOUT_MS,
                    socketTimeoutMS=MONGO_SOCKET_TIMEOUT_MS,
                    serverSelectionTimeoutMS=MONGO_CONNECT_TIMEOUT_MS,
                    maxPoolSize=MONGO_MAX_POOL_SIZE,
                    retryWrites=MONGO_RETRY_WRITES,
                    retryReads=MONGO_RETRY_READS
                )
                # Test the connection
                self.client.admin.command('ping')
                self.db = self.client[MONGO_DB]
                self.embeddings_collection = self.db[MONGO_EMBEDDINGS_COLLECTION]
                
                # Create index on assessment_id if it doesn't exist
                if "assessment_id_1" not in self.embeddings_collection.index_information():
                    self.embeddings_collection.create_index([("assessment_id", ASCENDING)])
                    logger.info(f"Created index on assessment_id in {MONGO_EMBEDDINGS_COLLECTION}")
                    
                logger.info(f"Connected to MongoDB collection: {MONGO_EMBEDDINGS_COLLECTION}")
                break  # Successfully connected, exit retry loop
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to connect to MongoDB: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Failed to connect to MongoDB after all retry attempts")
                    raise
        
        # Initialize Google Gemini embedding model
        try:
            # Test the model with a simple embedding request
            test_embedding = self.get_embedding("This is a test")
            self.embedding_dimension = len(test_embedding)
            logger.info(f"Successfully initialized Gemini embedding model. Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Error initializing Gemini embedding model: {e}")
            traceback.print_exc()
            
    def get_embedding(self, text):
        """Get embedding from Google Gemini API."""
        try:
            # Try using LangChain wrapper
            try:
                embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                return embedding_model.embed_query(text)
            except Exception as langchain_error:
                logger.warning(f"LangChain embedding failed: {langchain_error}, trying direct API")
                
                # Fall back to direct API call
                result = google_genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document")
                return result["embedding"]
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            traceback.print_exc()
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
            
    def process_assessments(self, assessments_df):
        """Process all assessments and store their embeddings."""
        if not self.client:
            logger.error("MongoDB client not initialized")
            return False
            
        try:
            logger.info(f"Processing embeddings for {len(assessments_df)} assessments")
            
            # Ensure text_for_embedding exists in dataframe
            if 'text_for_embedding' not in assessments_df.columns:
                logger.error("Column 'text_for_embedding' not found in dataframe")
                return False
                
            for idx, row in tqdm(assessments_df.iterrows(), total=len(assessments_df), desc="Processing embeddings"):
                # Extract assessment ID
                assessment_id = str(row.get('_id', idx))
                
                # Check if embedding already exists
                existing_embedding = self.get_stored_embedding(assessment_id)
                if existing_embedding:
                    logger.info(f"Embedding already exists for assessment ID: {assessment_id}")
                    continue
                    
                # Get text for embedding
                text = row['text_for_embedding']
                
                # Get embedding
                embedding = self.get_embedding(text)
                if not embedding:
                    logger.error(f"Failed to get embedding for assessment ID: {assessment_id}")
                    continue
                    
                # Store embedding
                self.store_embedding(assessment_id, text, embedding)
                
            logger.info("Finished processing embeddings")
            return True
        except Exception as e:
            logger.error(f"Error processing assessments: {e}")
            traceback.print_exc()
            return False
            
    def get_query_embedding(self, query_text):
        """Get embedding for a query."""
        return self.get_embedding(query_text)
        
    def search_similar(self, query_embedding, top_k=10):
        """Search for similar assessment embeddings using vector distance calculation in MongoDB."""
        if not self.client:
            logger.error("MongoDB client not initialized")
            return []
            
        try:
            # Convert query embedding to list if it's a numpy array
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
                
            # Calculate cosine similarity using MongoDB aggregation
            pipeline = [
                {
                    "$project": {
                        "assessment_id": 1,
                        "embedding": 1,
                        "similarity": {
                            "$let": {
                                "vars": {
                                    "dotProduct": {
                                        "$reduce": {
                                            "input": {"$zip": {"inputs": ["$embedding", query_embedding]}},
                                            "initialValue": 0,
                                            "in": {"$add": ["$$value", {"$multiply": [{"$arrayElemAt": ["$$this", 0]}, {"$arrayElemAt": ["$$this", 1]}]}]}
                                        }
                                    },
                                    "magnitude1": {
                                        "$sqrt": {
                                            "$reduce": {
                                                "input": "$embedding",
                                                "initialValue": 0,
                                                "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                                            }
                                        }
                                    },
                                    "magnitude2": {
                                        "$sqrt": {
                                            "$reduce": {
                                                "input": query_embedding,
                                                "initialValue": 0,
                                                "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                                            }
                                        }
                                    }
                                },
                                "in": {"$divide": ["$$dotProduct", {"$multiply": ["$$magnitude1", "$$magnitude2"]}]}
                            }
                        }
                    }
                },
                {"$sort": {"similarity": -1}},
                {"$limit": top_k}
            ]
            
            results = list(self.embeddings_collection.aggregate(pipeline))
            
            # Extract assessment IDs and similarity scores
            return [{"assessment_id": str(result["assessment_id"]), "similarity": result["similarity"]} for result in results]
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            traceback.print_exc()
            return []
            
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection") 