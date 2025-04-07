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

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "SHL")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "data")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class SHLRecommendationEngine:
    def __init__(self):
        print("Loading recommendation engine resources...")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        self.assessments_df = self.load_data_from_mongodb()
        
        if self.assessments_df is None or self.assessments_df.empty:
            print("Failed to load from MongoDB, falling back to JSON...")
            self.assessments_df = pd.read_json('data/processed_assessments.json')
        
        with open('data/embedding_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        self.index = faiss.read_index('data/assessment_index.faiss')
        self.initialize_llm()
        
        print("Recommendation engine ready!")
    
    def get_mongo_client(self):
        """Get MongoDB client connection."""
        client = MongoClient(MONGO_URI)
        return client
    
    def load_data_from_mongodb(self):
        """Load assessment data from MongoDB."""
        try:
            client = self.get_mongo_client()
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
    
    def recommend(self, query, url=None, top_k=10):
        """Recommend SHL assessments based on query/job description."""
        if url:
            url_content = self.fetch_content_from_url(url)
            if url_content:
                query = f"{query} {url_content}"
        
        enhanced_query = self.enhance_query(query)
        print(f"Enhanced Query: {enhanced_query[:100]}...")
        
        time_constraint = self.extract_duration_constraint(query)
        
        query_embedding = self.model.encode([enhanced_query])
        faiss.normalize_L2(query_embedding)
        
        k = min(top_k * 2, len(self.assessments_df))
        scores, indices = self.index.search(query_embedding, k)
        
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.assessments_df):
                assessment = self.assessments_df.iloc[idx].to_dict()
                assessment['score'] = float(scores[0][i])
                candidates.append(assessment)
        
        if time_constraint:
            candidates = [candidate for candidate in candidates if candidate.get('duration_minutes') and candidate['duration_minutes'] <= time_constraint]
        
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:top_k]
        
        results = []
        for candidate in candidates:
            result = {
                'name': candidate['name'],
                'url': candidate['url'],
                'remote_testing': candidate['remote_testing'],
                'adaptive_irt': candidate['adaptive_irt'],
                'duration': candidate['duration'],
                'test_type': candidate['test_type'],
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
        print(f"{i}. {rec['name']} ({rec['test_type']}) - {rec['duration']}")
        print(f"   Remote Testing: {rec['remote_testing']}, Adaptive: {rec['adaptive_irt']}")
        print(f"   URL: {rec['url']}")
        print()