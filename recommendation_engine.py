import pandas as pd
import numpy as np
import faiss
import pickle
import re
import requests
from bs4 import BeautifulSoup
import langchain_google_genai as genai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import logging
import traceback
import google.generativeai as google_genai
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
MONGO_CONNECT_TIMEOUT_MS = int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "30000"))
MONGO_SOCKET_TIMEOUT_MS = int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "45000"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class SHLRecommendationEngine:
    def __init__(self, skip_embedding_creation=False):
        print("Loading recommendation engine resources...")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        google_genai.configure(api_key=GOOGLE_API_KEY)
        
        # Track whether to skip embedding creation
        self.skip_embedding_creation = skip_embedding_creation
        if skip_embedding_creation:
            print("Skip embedding creation mode enabled - will use existing embeddings only")
        
        # Initialize Embedding Storage
        self.embedding_storage = EmbeddingStorage()
        
        # Load data
        self.assessments_df = self.load_data_from_mongodb()
        
        if self.assessments_df is None or self.assessments_df.empty:
            print("Failed to load from MongoDB, falling back to JSON...")
            self.assessments_df = self.load_data_from_json()
        
        # Prepare text for embedding if needed
        if 'text_for_embedding' not in self.assessments_df.columns:
            print("Preparing text for embedding...")
            self.assessments_df['text_for_embedding'] = self.assessments_df.apply(self.prepare_text_for_embedding, axis=1)
        
        # Skip automatic embedding processing to speed up initialization
        # Just check if embeddings exist instead
        print("Checking for existing embeddings...")
        try:
            # Check if there are embeddings in MongoDB
            client = self.get_mongo_client()
            db = client[MONGO_DB]
            embeddings_collection = db[MONGO_EMBEDDINGS_COLLECTION]
            embeddings_count = embeddings_collection.count_documents({})
            print(f"Found {embeddings_count} embeddings in MongoDB")
            client.close()
        except Exception as e:
            print(f"Error checking embeddings: {e}")
        
        # Load or create FAISS index (as fallback if MongoDB search is unavailable)
        self.load_or_create_index()
        
        # Initialize LLM for query enhancement
        self.initialize_llm()
        
        print("Recommendation engine ready!")
    
    def get_mongo_client(self):
        """Get MongoDB client connection."""
        client = MongoClient(
            MONGO_URI,
            connectTimeoutMS=MONGO_CONNECT_TIMEOUT_MS,
            socketTimeoutMS=MONGO_SOCKET_TIMEOUT_MS,
            serverSelectionTimeoutMS=MONGO_CONNECT_TIMEOUT_MS
        )
        return client
    
    def load_data_from_mongodb(self):
        """Load assessment data from MongoDB."""
        packaged_data = []
        individual_data = []
        
        try:
            client = self.get_mongo_client()
            db = client[MONGO_DB]
            
            # Load packaged solutions
            try:
                packaged_collection = db[MONGO_COLLECTION_PACKAGED]
                packaged_cursor = packaged_collection.find({})
                packaged_data = list(packaged_cursor)
            except Exception as e:
                print(f"Error loading packaged solutions: {e}")
            
            # Load individual solutions
            try:
                individual_collection = db[MONGO_COLLECTION_INDIVIDUAL]
                individual_cursor = individual_collection.find({})
                individual_data = list(individual_cursor)
            except Exception as e:
                print(f"Error loading individual solutions: {e}")
            
            # Combine data
            data = packaged_data + individual_data
            
            if not data:
                print("No data found in MongoDB collections")
                return None
            
            df = pd.DataFrame(data)
            
            print(f"Successfully loaded {len(df)} assessments from MongoDB")
            return df
        except Exception as e:
            print(f"Error loading from MongoDB: {e}")
            return None
        finally:
            client.close()
    
    def load_data_from_json(self):
        """Load assessment data from JSON file as fallback."""
        try:
            # Try to load processed assessments first
            if os.path.exists('data/processed_assessments.json'):
                df = pd.read_json('data/processed_assessments.json')
                print(f"Loaded {len(df)} assessments from processed_assessments.json")
                return df
            
            # Try original assessments
            if os.path.exists('data/shl_assessments.json'):
                df = pd.read_json('data/shl_assessments.json')
                print(f"Loaded {len(df)} assessments from shl_assessments.json")
                return df
            
            # Try sample data
            if os.path.exists('data/sample_assessments.json'):
                with open('data/sample_assessments.json', 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                print(f"Loaded {len(df)} assessments from sample_assessments.json")
                return df
            
            print("No assessment data files found")
            return None
        except Exception as e:
            print(f"Error loading JSON data: {e}")
            return None
    
    def load_or_create_index(self):
        """Load the FAISS index or create a new one if needed."""
        try:
            if os.path.exists('data/assessment_index.faiss'):
                self.index = faiss.read_index('data/assessment_index.faiss')
                print(f"Loaded FAISS index with {self.index.ntotal} vectors of dimension {self.index.d}")
                
                # Check if we have embeddings in the dataframe
                if 'embedding' not in self.assessments_df.columns and not self.skip_embedding_creation:
                    print("No embeddings found in dataframe. Creating embeddings...")
                    self.create_embeddings()
                    
                # Test a sample embedding to verify dimensions match
                if not self.skip_embedding_creation:
                    self.test_index_compatibility()
            else:
                if self.skip_embedding_creation:
                    print("No FAISS index found but skip_embedding_creation is enabled.")
                    print("Cannot proceed without either index or ability to create embeddings.")
                    raise ValueError("No FAISS index found with skip_embedding_creation enabled")
                else:
                    print("No FAISS index found. Creating embeddings and index...")
                    self.create_embeddings()
                    self.create_faiss_index()
        except Exception as e:
            if self.skip_embedding_creation:
                print(f"Error loading FAISS index: {e}")
                print("Cannot create embeddings because skip_embedding_creation is enabled.")
                raise ValueError("Failed to load index with skip_embedding_creation enabled") from e
            else:
                print(f"Error loading FAISS index: {e}")
                print("Creating new embeddings and index...")
                self.create_embeddings()
                self.create_faiss_index()
    
    def test_index_compatibility(self):
        """Test compatibility between the loaded index and the embeddings."""
        try:
            if 'embedding' not in self.assessments_df.columns:
                print("No embeddings found in dataframe. Cannot test compatibility.")
                return
                
            # Get a sample embedding from the dataframe
            sample_embedding = self.assessments_df['embedding'].iloc[0]
            
            # Check if the sample embedding is in the index
            if sample_embedding not in self.index:
                print("Warning: Sample embedding not found in the index.")
            
            # Test embedding compatibility
            if not self.skip_embedding_creation:
                self.create_embeddings()
                self.test_index_compatibility()
        except Exception as e:
            print(f"Error testing index compatibility: {e}")
            traceback.print_exc()
    
    def create_embeddings(self):
        """Create embeddings for assessments."""
        print("Creating embeddings for assessments...")
        
        # Generate embeddings
        embeddings = []
        
        for idx, row in enumerate(self.assessments_df.iterrows()):
            text = row[1]['text_for_embedding']
            embedding = self.get_embedding(text)
            if embedding:
                embeddings.append(embedding)
            else:
                # If embedding fails, add a zero vector of the same dimension
                if embeddings:
                    embeddings.append([0.0] * len(embeddings[0]))
                else:
                    # If this is the first item and it failed, add a placeholder
                    embeddings.append([0.0])
            
            if idx % 10 == 0:
                print(f"Created {idx}/{len(self.assessments_df)} embeddings...")
                
        self.assessments_df['embedding'] = embeddings
        print(f"Created {len(embeddings)} embeddings")
    
    def create_faiss_index(self):
        """Create a FAISS index for fast vector similarity search."""
        try:
            if 'embedding' not in self.assessments_df.columns:
                print("No embeddings found in dataframe. Cannot create FAISS index.")
                return
                
            embeddings = np.vstack(self.assessments_df['embedding'].values)
            embedding_matrix = np.array(embeddings).astype('float32')
            dimension = embedding_matrix.shape[1]
            
            print(f"Creating FAISS index with {len(embeddings)} embeddings of dimension {dimension}")
            
            self.index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embedding_matrix)
            self.index.add(embedding_matrix)
            
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            faiss.write_index(self.index, 'data/assessment_index.faiss')
            print(f"Created and saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            traceback.print_exc()
    
    def initialize_llm(self):
        """Initialize the Google Gemini model for query enhancement."""
        self.llm = genai.ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        
        self.query_template = PromptTemplate(
            input_variables=["query"],
            template="""You are a job assessment specialist. Enhance the following job description or query to include key skills, competencies, and assessment needs relevant for matching with SHL assessments. Original Query: {query} Enhanced Query:"""
        )
        
        self.query_chain = self.query_template | self.llm | StrOutputParser()
    
    def enhance_query(self, query):
        """Use LLM to enhance the query with relevant skills and context."""
        try:
            enhanced_query = self.query_chain.invoke({"query": query})
            return enhanced_query
        except Exception as e:
            print(f"Error enhancing query: {e}")
            return query
    
    def extract_duration_constraint(self, query):
        """Extract any time/duration constraint from the query."""
        time_patterns = [
            r"(\d+)\s*minutes",
            r"(\d+)\s*mins",
            r"within\s*(\d+)",
            r"less than\s*(\d+)",
            r"max.*?(\d+)",
            r"maximum.*?(\d+)"
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                try:
                    return int(matches[0])
                except:
                    pass
        
        return None
    
    def extract_skills(self, query):
        """Extract key skills mentioned in the query."""
        tech_skills = ["python", "java", "javascript", "sql", "c++", "c#", "ruby", "php"]
        soft_skills = ["communication", "leadership", "teamwork", "collaboration", "problem solving"]
        
        mentioned_skills = []
        for skill in tech_skills + soft_skills:
            if skill.lower() in query.lower():
                mentioned_skills.append(skill)
        
        return mentioned_skills
    
    def fetch_content_from_url(self, url):
        """Fetch and extract content from a job description URL."""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            content = soup.find('main') or soup.find('article') or soup.find('body')
            if content:
                return content.get_text(separator=' ', strip=True)
            return ""
        except Exception as e:
            print(f"Error fetching URL: {e}")
            return ""
    
    def get_embedding(self, text):
        """Get embedding for text using Gemini."""
        # First try to use our embedding storage
        embedding = self.embedding_storage.get_embedding(text)
        if embedding:
            return embedding
            
        # Fallback to direct Gemini API call
        try:
            print("Trying direct Gemini API as fallback...")
            result = google_genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document")
            embedding = result["embedding"]
            return embedding
        except Exception as fallback_error:
            print(f"Fallback embedding also failed: {fallback_error}")
            return None
    
    def prepare_text_for_embedding(self, row):
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
    
    def recommend(self, query, url=None, top_k=10):
        """
        Recommend SHL assessments based on a query or URL.
        """
        try:
            # Process URL if provided
            if url:
                content = self.fetch_content_from_url(url)
                if content:
                    query = content if not query else f"{query}\n\nJob Description: {content}"
            
            # Enhance query with Gemini
            enhanced_query = self.enhance_query(query)
            print(f"Enhanced query: {enhanced_query[:100]}...")
            
            # Extract duration constraint
            duration_constraint = self.extract_duration_constraint(query)
            print(f"Duration constraint: {duration_constraint} minutes")
            
            # Extract skills
            skills = self.extract_skills(query)
            print(f"Extracted skills: {skills}")
            
            # Get embedding for query
            query_embedding = self.embedding_storage.get_query_embedding(enhanced_query)
            
            # Try MongoDB-based similarity search first
            try:
                similar_results = self.embedding_storage.search_similar(query_embedding, top_k=top_k*2)
                
                if similar_results:
                    print(f"Found {len(similar_results)} similar assessments using MongoDB")
                    
                    # Get assessment details from the dataframe
                    recommendations = []
                    for result in similar_results:
                        assessment_id = result["assessment_id"]
                        similarity = result["similarity"]
                        
                        # Find the assessment in the dataframe
                        matching_rows = self.assessments_df[self.assessments_df['_id'].astype(str) == assessment_id]
                        
                        if not matching_rows.empty:
                            assessment = matching_rows.iloc[0].to_dict()
                            assessment['similarity_score'] = similarity
                            recommendations.append(assessment)
                    
                    # Apply duration constraint if specified
                    if duration_constraint and duration_constraint > 0:
                        filtered_recommendations = []
                        for rec in recommendations:
                            # Check if duration is within the constraint
                            duration = rec.get('duration_minutes', 0)
                            if duration <= duration_constraint:
                                filtered_recommendations.append(rec)
                        
                        if filtered_recommendations:
                            recommendations = filtered_recommendations
                        else:
                            print(f"No recommendations match the duration constraint of {duration_constraint} minutes")
                    
                    # Sort by similarity score
                    recommendations = sorted(recommendations, key=lambda x: x.get('similarity_score', 0), reverse=True)
                    
                    # Return top_k recommendations
                    return recommendations[:top_k]
            except Exception as e:
                print(f"Error in MongoDB similarity search: {e}")
                traceback.print_exc()
            
            # Fallback to FAISS if MongoDB search fails
            print("Falling back to FAISS index for search...")
            
            # Get embedding for query
            embedding = self.get_embedding(enhanced_query)
            if embedding is None:
                return []
            
            # Convert to numpy array
            query_vector = np.array([embedding], dtype=np.float32)
            
            # Search
            distances, indices = self.index.search(query_vector, top_k * 2)
            
            # Get recommendations
            recommendations = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.assessments_df):
                    assessment = self.assessments_df.iloc[idx].to_dict()
                    assessment['similarity_score'] = float(1.0 - distances[0][i])
                    recommendations.append(assessment)
            
            # Apply duration constraint if specified
            if duration_constraint and duration_constraint > 0:
                filtered_recommendations = []
                for rec in recommendations:
                    duration = rec.get('duration_minutes', 0)
                    if duration <= duration_constraint:
                        filtered_recommendations.append(rec)
                
                if filtered_recommendations:
                    recommendations = filtered_recommendations
            
            # Sort by similarity score
            recommendations = sorted(recommendations, key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            return recommendations[:top_k]
        except Exception as e:
            print(f"Error in recommendation: {e}")
            traceback.print_exc()
            return []
    
    def generate_natural_language_response(self, query, recommendations, duration_constraint=None):
        """
        Generate a natural language response using Gemini based on the query and recommendations.
        
        Args:
            query (str): The original user query
            recommendations (list): List of recommended assessments
            duration_constraint (int, optional): Any duration constraint extracted from the query
            
        Returns:
            str: Natural language response from Gemini
        """
        try:
            # Check if Google Gemini API key is available
            if not GOOGLE_API_KEY:
                return "Natural language response unavailable (Gemini API key not configured)."
            
            # Create prompt for Gemini
            prompt_template = """
            You are an SHL Assessment Specialist helping a hiring manager find the right assessments.
            
            User Query: {query}
            
            Top Recommended Assessments:
            {formatted_recommendations}
            
            {duration_info}
            
            Based on the user's query and the recommendations, provide a helpful analysis that:
            1. Identifies the key skills and requirements from the query
            2. Explains why these specific assessments are recommended
            3. Highlights how they align with the job requirements
            4. Provides guidance on how to use these assessments in the hiring process
            
            Keep your response concise (about 150-200 words) and conversational.
            """
            
            # Format the recommendations for the prompt
            formatted_recs = []
            for i, rec in enumerate(recommendations[:5], 1):  # Use top 5 for the analysis
                formatted_recs.append(f"{i}. {rec['name']} ({rec['test_type']}, Duration: {rec['duration']})")
            
            formatted_recommendations = "\n".join(formatted_recs)
            
            # Add duration constraint info if available
            duration_info = ""
            if duration_constraint:
                duration_info = f"Note: The user specified a duration constraint of {duration_constraint} minutes."
            
            # Prepare the prompt
            prompt = prompt_template.format(
                query=query,
                formatted_recommendations=formatted_recommendations,
                duration_info=duration_info
            )
            
            # Initialize Gemini model
            model = genai.ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
            
            # Generate the response
            response = model.invoke(prompt)
            
            # Extract the content
            if hasattr(response, 'content') and response.content:
                return response.content
            else:
                logger.warning("Empty response from Gemini")
                return "I couldn't generate a detailed analysis at this time. Please review the recommended assessments above."
                
        except Exception as e:
            logger.error(f"Error generating natural language response: {str(e)}")
            logger.error(traceback.format_exc())
            return f"I couldn't generate a detailed analysis at this time. Please review the recommended assessments above."
    
    def __del__(self):
        """Clean up resources."""
        try:
            self.embedding_storage.close()
        except:
            pass

if __name__ == "__main__":
    engine = SHLRecommendationEngine()
    
    test_query = "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment that can be completed in 40 minutes."
    
    recommendations = engine.recommend(test_query, top_k=5)
    
    print("Test Query:", test_query)
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} ({', '.join(rec['test_type_names'])}) - {rec['duration']}")
        print(f"   Remote Testing: {rec['remote_testing']}, Adaptive: {rec['adaptive_irt']}")
        print(f"   URL: {rec['url']}")
        print(f"   Score: {rec['score']}")
        print()