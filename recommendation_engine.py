import pandas as pd
import numpy as np
import faiss
import pickle
import re
import requests
from bs4 import BeautifulSoup
import langchain_google_genai as genai
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import logging
import traceback
import google.generativeai as google_genai
from embedding_storage import EmbeddingStorage  
import json
from typing import Optional, List

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
            
            # Load embeddings from MongoDB
            try:
                embeddings_collection = db[MONGO_EMBEDDINGS_COLLECTION]
                embeddings_data = {}
                for doc in embeddings_collection.find():
                    embeddings_data[doc['assessment_id']] = doc['embedding']
                
                # Add embeddings to dataframe
                df['embedding'] = df['_id'].astype(str).map(embeddings_data)
                print(f"Loaded {len(embeddings_data)} embeddings from MongoDB")
            except Exception as e:
                print(f"Error loading embeddings: {e}")
            
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
                
                # Check if dimensions match
                if self.index.d != 768:  # Gemini embedding dimension
                    print("Index dimension mismatch. Recreating index...")
                    self.create_faiss_index()
            else:
                print("No FAISS index found. Creating new index...")
                self.create_faiss_index()
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            print("Creating new index...")
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
            
            # Create index with correct dimension
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
        self.llm = genai.ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
        self.query_template = PromptTemplate(
            input_variables=["query"],
            template="""You are a job assessment specialist. Enhance the following job description or query to include key skills, competencies, and assessment needs relevant for matching with SHL assessments. Focus on:

1. Job level (entry-level, mid-level, senior, etc.)
2. Specific role requirements (managerial, technical, customer service, etc.)
3. Required skills and competencies
4. Assessment preferences (duration, test types, etc.)
5. Language requirements

Original Query: {query}

Enhanced Query:"""
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
    
    def understand_query_with_gemini(self, query):
        """Use Gemini to understand the query and extract structured information."""
        try:
            # Define function for Gemini to call
            query_analysis_fn = {
                "name": "analyze_query",
                "description": "Analyze a job assessment query to extract key information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "duration_minutes": {
                            "type": "integer",
                            "description": "Duration constraint in minutes, if specified in the query"
                        },
                        "skills": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of skills mentioned or required in the query"
                        },
                        "test_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of test types mentioned or preferred in the query"
                        }
                    },
                    "required": ["skills", "test_types"]
                }
            }

            model = genai.ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.1
            )
            
            # Create messages for the model
            messages = [
                {"role": "system", "content": """You are an expert at understanding job assessment queries. Analyze the following query and extract:
                1. Duration constraints (in minutes)
                2. Required skills
                3. Test type preferences
                
                Return the information in a structured format."""},
                {"role": "user", "content": query}
            ]
            
            response = model.invoke(
                messages, 
                tools=[query_analysis_fn]
            )
            
            # Extract function call result
            if response.tool_calls and len(response.tool_calls) > 0:
                tool_call = response.tool_calls[0]
                if tool_call.function.name == "analyze_query":
                    import json
                    result = json.loads(tool_call.function.arguments)
                    return {
                        "duration_minutes": result.get("duration_minutes"),
                        "skills": result.get("skills", []),
                        "test_types": result.get("test_types", [])
                    }
            
            # Fallback if no function call was made
            return {"duration_minutes": None, "skills": [], "test_types": []}
        except Exception as e:
            print(f"Error using Gemini for query understanding: {e}")
            traceback.print_exc()
            return {"duration_minutes": None, "skills": [], "test_types": []}
    
    def extract_duration_constraint(self, query):
        """Extract any time/duration constraint from the query using Gemini."""
        try:
            # Get structured understanding from Gemini
            query_info = self.understand_query_with_gemini(query)
            
            # Extract duration if available
            duration = query_info.get("duration_minutes")
            if duration is not None:
                print(f"Gemini detected duration: {duration} minutes")
                return duration
            
            # Fallback to regex patterns if Gemini didn't find duration
            time_patterns = [
                # Hour-based patterns (check these first)
                r'(\d+)\s*hour',
                r'(\d+)\s*hr',
                r'less than\s*(\d+)\s*hour',
                r'under\s*(\d+)\s*hour',
                r'no more than\s*(\d+)\s*hour',
                r'up to\s*(\d+)\s*hour',
                r'(\d+)\s*hour or less',
                r'no longer than\s*(\d+)\s*hour',
                r'(\d+)\s*hour maximum',
                
                # Minute-based patterns (check these second)
                r'(\d+)\s*(?:minute|min)',
                r'(\d+)\s*mins',
                r'within\s*(\d+)',
                r'less than\s*(\d+)',
                r'max.*?(\d+)',
                r'maximum.*?(\d+)',
                r'under (\d+)\s*(?:minute|min)',
                r'shorter than (\d+)\s*(?:minute|min)',
                r'(\d+)\s*(?:minute|min) or less',
                r'no longer than (\d+)\s*(?:minute|min)',
                r'no more than (\d+)\s*(?:minute|min)',
                r'(\d+)\s*(?:minute|min) maximum',
                r'up to (\d+)\s*(?:minute|min)'
            ]
            
            # Check hour patterns first (first 9 patterns)
            for i, pattern in enumerate(time_patterns):
                matches = re.findall(pattern, query.lower())
                if matches:
                    try:
                        duration = int(matches[0])
                        # Convert hours to minutes if the pattern was hour-based (first 9 patterns)
                        if i < 9:  # First 9 patterns are hour-based
                            print(f"Found hour-based duration: {duration} hours")
                            duration *= 60
                        else:
                            print(f"Found minute-based duration: {duration} minutes")
                        return duration
                    except:
                        pass
            
            return None
        except Exception as e:
            print(f"Error extracting duration constraint: {e}")
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
        # Helper function to safely convert to list
        def safe_to_list(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return value
            if isinstance(value, str):
                return [value]
            return [str(value)]

        fields = [
            str(row.get('name', '')),
            str(row.get('description', '')),
            str(row.get('category', '')),
            ' '.join(safe_to_list(row.get('job_levels'))),
            ' '.join(safe_to_list(row.get('test_type_names'))),
            ' '.join(safe_to_list(row.get('languages'))),
            str(row.get('duration_minutes', '')),
            str(row.get('adaptive_irt', '')),
            str(row.get('remote_testing', ''))
        ]
        
        # Add specific role-related keywords if present in name or description
        role_keywords = ['manager', 'supervisor', 'lead', 'director', 'coordinator', 'specialist', 'analyst', 'developer']
        for keyword in role_keywords:
            if keyword in str(row.get('name', '')).lower() or keyword in str(row.get('description', '')).lower():
                fields.append(keyword)
        
        # Add test type keywords
        test_type_keywords = ['personality', 'cognitive', 'skills', 'knowledge', 'behavioral', 'situational', 'judgment']
        for keyword in test_type_keywords:
            if keyword in ' '.join(safe_to_list(row.get('test_type_names'))).lower():
                fields.append(keyword)
        
        # Combine all fields with proper spacing
        text = ' '.join(str(field) for field in fields if field)
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        return text
    
    def recommend(self, query, url=None, top_k=10):
        """
        Recommend SHL assessments based on a query or URL.
        
        Args:
            query (str): The user's search query
            url (str, optional): URL to job posting to extract additional context
            top_k (int): Number of recommendations to return
            
        Returns:
            list: List of recommendation dictionaries containing assessment details and scores
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
            if duration_constraint:
                print(f"Duration constraint: {duration_constraint} minutes")
            
            # Extract skills
            skills = self.extract_skills(query)
            print(f"Extracted skills: {skills}")
            
            # Get embedding for query
            query_embedding = self.get_embedding(enhanced_query)
            
            # Try MongoDB-based similarity search first
            try:
                similar_results = self.embedding_storage.search_similar(query_embedding, top_k=top_k*2)
                
                if similar_results:
                    print(f"Found {len(similar_results)} similar assessments using MongoDB")
                    
                    # Get assessment details from the dataframe
                    recommendations = []
                    for result in similar_results:
                        assessment_id = result["assessment_id"]
                        base_similarity = result["similarity"]
                        
                        # Find the assessment in the dataframe
                        matching_rows = self.assessments_df[self.assessments_df['_id'].astype(str) == assessment_id]
                        
                        if not matching_rows.empty:
                            assessment = matching_rows.iloc[0].to_dict()
                            
                            # Calculate boosted score based on multiple factors
                            score = base_similarity
                            
                            # Boost score if skills match
                            if skills:
                                assessment_skills = assessment.get('skills', [])
                                if isinstance(assessment_skills, str):
                                    assessment_skills = [s.strip() for s in assessment_skills.split(',')]
                                skill_matches = sum(1 for skill in skills if skill.lower() in [s.lower() for s in assessment_skills])
                                if skill_matches > 0:
                                    score *= (1 + (skill_matches * 0.2))  # Boost by 20% per matching skill
                            
                            # Boost score if duration matches constraint
                            if duration_constraint:
                                assessment_duration = assessment.get('duration_minutes', 0)
                                if assessment_duration <= duration_constraint:
                                    score *= 1.3  # Boost by 30% if duration matches
                            
                            # Boost score if test type matches
                            test_type = assessment.get('test_type', '')
                            if test_type:
                                # Check if test_type is a list or string and handle accordingly
                                if isinstance(test_type, list):
                                    # If it's a list, check if any of our target types are in the list
                                    if any(any(target.lower() in t.lower() for t in test_type if isinstance(t, str)) 
                                           for target in ['cognitive', 'skills', 'personality']):
                                        score *= 1.2  # Boost by 20% for relevant test types
                                else:
                                    # If it's not a list (assumed to be string), proceed as before
                                    if any(t.lower() in test_type.lower() for t in ['cognitive', 'skills', 'personality']):
                                        score *= 1.2  # Boost by 20% for relevant test types
                            
                            # Normalize score to be between 0 and 1
                            score = min(1.0, max(0.0, score))
                            
                            # Convert numpy/pandas types to Python native types for JSON serialization
                            for key, value in assessment.items():
                                if isinstance(value, (np.integer, np.floating)):
                                    assessment[key] = float(value) if isinstance(value, np.floating) else int(value)
                                elif isinstance(value, np.ndarray):
                                    assessment[key] = value.tolist()
                            
                            # Add score for API format compatibility
                            assessment['score'] = float(score)
                            recommendations.append(assessment)
                    
                    # Apply duration constraint if specified
                    if duration_constraint and duration_constraint > 0:
                        original_count = len(recommendations)
                        filtered_recommendations = []
                        for rec in recommendations:
                            duration = rec.get('duration_minutes', 0)
                            if duration <= duration_constraint:
                                filtered_recommendations.append(rec)
                        
                        if filtered_recommendations:
                            print(f"Filtered from {original_count} to {len(filtered_recommendations)} recommendations based on duration constraint")
                            recommendations = filtered_recommendations
                    
                    # Sort by boosted score
                    recommendations = sorted(recommendations, key=lambda x: x.get('score', 0), reverse=True)
                    
                    # Format for API response
                    for rec in recommendations:
                        # Ensure all values can be JSON serialized
                        for key, value in list(rec.items()):
                            if isinstance(value, (np.ndarray, list)) and key != 'test_type_names' and key != 'job_levels':
                                rec[key] = str(value)
                    
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
                    
                    # Calculate boosted score based on multiple factors
                    base_similarity = float(1.0 - distances[0][i])
                    score = base_similarity
                    
                    # Boost score if skills match
                    if skills:
                        assessment_skills = assessment.get('skills', [])
                        if isinstance(assessment_skills, str):
                            assessment_skills = [s.strip() for s in assessment_skills.split(',')]
                        skill_matches = sum(1 for skill in skills if skill.lower() in [s.lower() for s in assessment_skills])
                        if skill_matches > 0:
                            score *= (1 + (skill_matches * 0.2))  # Boost by 20% per matching skill
                    
                    # Boost score if duration matches constraint
                    if duration_constraint:
                        assessment_duration = assessment.get('duration_minutes', 0)
                        if assessment_duration <= duration_constraint:
                            score *= 1.3  # Boost by 30% if duration matches
                    
                    # Boost score if test type matches
                    test_type = assessment.get('test_type', '')
                    if test_type:
                        # Check if test_type is a list or string and handle accordingly
                        if isinstance(test_type, list):
                            # If it's a list, check if any of our target types are in the list
                            if any(any(target.lower() in t.lower() for t in test_type if isinstance(t, str)) 
                                   for target in ['cognitive', 'skills', 'personality']):
                                score *= 1.2  # Boost by 20% for relevant test types
                        else:
                            # If it's not a list (assumed to be string), proceed as before
                            if any(t.lower() in test_type.lower() for t in ['cognitive', 'skills', 'personality']):
                                score *= 1.2  # Boost by 20% for relevant test types
                    
                    # Normalize score to be between 0 and 1
                    score = min(1.0, max(0.0, score))
                    
                    # Convert numpy/pandas types to Python native types for JSON serialization
                    for key, value in assessment.items():
                        if isinstance(value, (np.integer, np.floating)):
                            assessment[key] = float(value) if isinstance(value, np.floating) else int(value)
                        elif isinstance(value, np.ndarray):
                            assessment[key] = value.tolist()
                    
                    # Add score for API format compatibility
                    assessment['score'] = float(score)
                    recommendations.append(assessment)
            
            # Apply duration constraint if specified
            if duration_constraint and duration_constraint > 0:
                original_count = len(recommendations)
                filtered_recommendations = []
                for rec in recommendations:
                    duration = rec.get('duration_minutes', 0)
                    if duration <= duration_constraint:
                        filtered_recommendations.append(rec)
                
                if filtered_recommendations:
                    print(f"Filtered from {original_count} to {len(filtered_recommendations)} recommendations based on duration constraint")
                    recommendations = filtered_recommendations
            
            # Sort by boosted score
            recommendations = sorted(recommendations, key=lambda x: x.get('score', 0), reverse=True)
            
            # Format for API response
            for rec in recommendations:
                # Ensure all values can be JSON serialized
                for key, value in list(rec.items()):
                    if isinstance(value, (np.ndarray, list)) and key != 'test_type_names' and key != 'job_levels':
                        rec[key] = str(value)
            
            return recommendations[:top_k]
        except Exception as e:
            print(f"Error in recommendation: {e}")
            traceback.print_exc()
            return []
    
    def generate_natural_language_response(self, query, recommendations, duration_constraint=None):
        """Generate a natural language response using Gemini based on the query and recommendations."""
        try:
            if not GOOGLE_API_KEY:
                return "Natural language response unavailable (Gemini API key not configured)."
            
            # Format recommendations for the prompt
            formatted_recs = []
            for i, rec in enumerate(recommendations[:5], 1):
                name = rec.get('name', 'Unknown Assessment')
                test_type = rec.get('test_type', '')
                if not test_type and 'test_type_names' in rec:
                    test_type = ', '.join(rec['test_type_names']) if isinstance(rec['test_type_names'], list) else rec['test_type_names']
                duration = rec.get('duration_minutes', 'Unknown duration')
                if duration != 'Unknown duration':
                    duration = f"{duration} minutes"
                formatted_recs.append(f"{i}. {name} ({test_type}, Duration: {duration})")
            
            formatted_recommendations = "\n".join(formatted_recs)
            duration_info = f"Note: The user specified a duration constraint of {duration_constraint} minutes." if duration_constraint else ""
            
            # Define function for Gemini to call
            response_fn = {
                "name": "generate_response",
                "description": "Generate a natural language response to assessment recommendations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis": {
                            "type": "string",
                            "description": "Detailed analysis of the recommendations"
                        },
                        "key_points": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key points about the recommendations"
                        },
                        "next_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Suggested next steps for the user"
                        }
                    },
                    "required": ["analysis"]
                }
            }
            
            model = genai.ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7
            )
            
            # Create messages for the model
            messages = [
                {"role": "system", "content": """You are an SHL Assessment Specialist helping a hiring manager find the right assessments.
                
                Based on the user's query and the recommendations, provide a helpful analysis that:
                1. Identifies the key skills and requirements from the query
                2. Explains why these specific assessments are recommended
                3. Highlights how they align with the job requirements
                4. Provides guidance on how to use these assessments in the hiring process
                
                Keep your response concise (about 150-200 words) and conversational.
                Always mention the duration of each assessment in your analysis."""},
                {"role": "user", "content": f"""User Query: {query}
                
                Top Recommended Assessments:
                {formatted_recommendations}
                
                {duration_info}"""}
            ]
            
            response = model.invoke(
                messages,
                tools=[response_fn]
            )
            
            # Extract function call result
            if response.tool_calls and len(response.tool_calls) > 0:
                tool_call = response.tool_calls[0]
                if tool_call.function.name == "generate_response":
                    import json
                    result = json.loads(tool_call.function.arguments)
                    return result.get("analysis", "")
            
            # Fallback to direct text response if function calling failed
            if hasattr(response, 'content'):
                return str(response.content)
            elif hasattr(response, 'text'):
                return str(response.text)
                
            return "I couldn't generate a detailed analysis at this time. Please review the recommended assessments above."
            
        except Exception as e:
            logger.error(f"Error generating natural language response: {str(e)}")
            logger.error(traceback.format_exc())
            return "I couldn't generate a detailed analysis at this time. Please review the recommended assessments above."
    
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
        print(f"{i}. {rec['name']} ({rec.get('test_type', '')}) - {rec.get('duration', 'Unknown')}")
        print(f"   Remote Testing: {rec.get('remote_testing', 'Unknown')}, Adaptive: {rec.get('adaptive_irt', 'Unknown')}")
        print(f"   URL: {rec.get('url', '#')}")
        print(f"   Score: {rec.get('score', 0)}")
        print()