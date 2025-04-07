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
from openai import AzureOpenAI
import logging
import traceback

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
MONGO_CONNECT_TIMEOUT_MS = int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "30000"))
MONGO_SOCKET_TIMEOUT_MS = int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "45000"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Azure OpenAI connection
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o")

class SHLRecommendationEngine:
    def __init__(self):
        print("Loading recommendation engine resources...")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        
        # Initialize Azure OpenAI client
        try:
            self.azure_client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
        except Exception as e:
            print(f"Warning: Could not initialize Azure OpenAI client: {e}")
            self.azure_client = None
        
        # Load data
        self.assessments_df = self.load_data_from_mongodb()
        
        if self.assessments_df is None or self.assessments_df.empty:
            print("Failed to load from MongoDB, falling back to JSON...")
            self.assessments_df = self.load_data_from_json()
        
        # Check if we have saved Azure info
        try:
            with open('data/azure_openai_info.pkl', 'rb') as f:
                self.azure_info = pickle.load(f)
                print(f"Loaded Azure OpenAI info for {self.azure_info['embedding_model']}")
        except FileNotFoundError:
            self.azure_info = {
                "api_version": AZURE_OPENAI_API_VERSION,
                "embedding_model": AZURE_EMBEDDING_DEPLOYMENT_NAME,
                "endpoint": AZURE_OPENAI_ENDPOINT
            }
            print("Created new Azure OpenAI info configuration")
        
        # Load or create FAISS index
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
                if 'embedding' not in self.assessments_df.columns:
                    print("No embeddings found in dataframe. Creating embeddings...")
                    self.create_embeddings()
                    
                # Test a sample embedding to verify dimensions match
                self.test_index_compatibility()
            else:
                print("No FAISS index found. Creating embeddings and index...")
                self.create_embeddings()
                self.create_faiss_index()
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            print("Creating new embeddings and index...")
            self.create_embeddings()
            self.create_faiss_index()
    
    def test_index_compatibility(self):
        """Test compatibility between Azure embeddings and the loaded index."""
        try:
            if self.azure_client is None:
                print("Azure OpenAI client not initialized. Skipping compatibility test.")
                return
                
            sample_text = "Test query for Java developers"
            
            # Get embedding from Azure OpenAI
            embedding = self.get_embedding(sample_text)
            embedding_array = np.array([embedding], dtype=np.float32)
            
            # Check dimensions
            expected_dim = self.index.d
            actual_dim = embedding_array.shape[1]
            
            print(f"Testing index compatibility: expected dim={expected_dim}, actual dim={actual_dim}")
            
            if expected_dim != actual_dim:
                print("Dimension mismatch! Rebuilding index...")
                self.create_embeddings()
                self.create_faiss_index()
            else:
                print("Index dimensions are compatible")
                
        except Exception as e:
            print(f"Error testing index compatibility: {e}")
            traceback.print_exc()
    
    def create_embeddings(self):
        """Create embeddings for assessments."""
        if self.azure_client is None:
            print("Azure OpenAI client not initialized. Cannot create embeddings.")
            return
            
        print("Creating embeddings for assessments...")
        
        # Prepare text for embedding
        self.assessments_df['text_for_embedding'] = self.assessments_df.apply(self.prepare_text_for_embedding, axis=1)
        
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
        """Get embedding from Azure OpenAI API."""
        try:
            if self.azure_client is None:
                print("Azure OpenAI client not initialized. Cannot get embedding.")
                return None
                
            response = self.azure_client.embeddings.create(
                input=text,
                model=AZURE_EMBEDDING_DEPLOYMENT_NAME
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            traceback.print_exc()
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
        """Recommend SHL assessments based on query/job description."""
        if url:
            url_content = self.fetch_content_from_url(url)
            if url_content:
                query = f"{query} {url_content}"
        
        # Enhance query with LLM
        enhanced_query = self.enhance_query(query)
        print(f"Enhanced Query: {enhanced_query[:100]}...")
        
        # Extract any time constraints
        time_constraint = self.extract_duration_constraint(query)
        
        # Prepare query for embedding
        prepared_query = self.prepare_text_for_embedding({"name": enhanced_query})
        
        # Get embedding from Azure OpenAI
        query_embedding_raw = self.get_embedding(prepared_query)
        if query_embedding_raw is None:
            print("Error getting query embedding")
            return []
            
        query_embedding = np.array([query_embedding_raw], dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search for similar assessments
        k = min(top_k * 2, len(self.assessments_df))
        
        # Check dimensions match
        if self.index.d != query_embedding.shape[1]:
            print(f"Error: dimension mismatch. Index dim={self.index.d}, query dim={query_embedding.shape[1]}")
            return []
            
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare candidates
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.assessments_df):
                assessment = self.assessments_df.iloc[idx].to_dict()
                assessment['score'] = float(scores[0][i])
                candidates.append(assessment)
        
        # Filter by time constraint if provided
        if time_constraint:
            candidates = [candidate for candidate in candidates if candidate.get('duration_minutes') and candidate['duration_minutes'] <= time_constraint]
        
        # Sort by score and limit to top_k
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:top_k]
        
        # Format results
        results = []
        for candidate in candidates:
            result = {
                'name': candidate['name'],
                'url': candidate.get('url', '#'),
                'remote_testing': candidate.get('remote_testing', False),
                'adaptive_irt': candidate.get('adaptive_irt', False),
                'duration': candidate.get('assessment_length', 'Unknown'),
                'duration_minutes': candidate.get('duration_minutes', 0),
                'test_type': candidate.get('test_type', []),
                'test_type_names': candidate.get('test_type_names', []),
                'score': round(candidate['score'], 3)
            }
            results.append(result)
        
        return results

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