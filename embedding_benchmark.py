import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import google.generativeai as google_genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
google_genai.configure(api_key=GOOGLE_API_KEY)

def get_sentence_transformer_model(model_name="all-MiniLM-L6-v2"):
    """Initialize and return SentenceTransformer model."""
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        logger.error(f"Error initializing SentenceTransformer model: {e}")
        return None

def get_sentence_embedding(text, model):
    """Get embedding from SentenceTransformer model."""
    try:
        embedding = model.encode([text])[0]
        return embedding
    except Exception as e:
        logger.error(f"Error getting SentenceTransformer embedding: {e}")
        return None

def get_gemini_embedding(text):
    """Get embedding from Gemini API."""
    try:
        # Try with LangChain first
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
        logger.error(f"Error getting Gemini embedding: {e}")
        return None

def benchmark_embeddings(texts, num_iterations=3):
    """Run embedding benchmark on different models."""
    results = {
        "method": [],
        "dimension": [],
        "time_ms": [],
        "success": []
    }
    
    # Initialize models
    sentence_model = get_sentence_transformer_model()
            
    if not sentence_model:
        logger.error("SentenceTransformer model initialization failed. Skipping tests.")
        return pd.DataFrame(results)
    
    logger.info("Running benchmark...")
    
    # Test SentenceTransformer embeddings
    for i in range(num_iterations):
        for text in texts:
            start_time = time.time()
            embedding = get_sentence_embedding(text, sentence_model)
            end_time = time.time()
            
            time_ms = (end_time - start_time) * 1000
            success = embedding is not None
            dimension = len(embedding) if success else 0
            
            results["method"].append("SentenceTransformer")
            results["dimension"].append(dimension)
            results["time_ms"].append(time_ms)
            results["success"].append(success)
            
            logger.info(f"SentenceTransformer embedding: dim={dimension}, time={time_ms:.2f}ms, success={success}")
    
    # Test Gemini embeddings
    for i in range(num_iterations):
        for text in texts:
            start_time = time.time()
            embedding = get_gemini_embedding(text)
            end_time = time.time()
            
            time_ms = (end_time - start_time) * 1000
            success = embedding is not None
            dimension = len(embedding) if success else 0
            
            results["method"].append("Gemini")
            results["dimension"].append(dimension)
            results["time_ms"].append(time_ms)
            results["success"].append(success)
            
            logger.info(f"Gemini embedding: dim={dimension}, time={time_ms:.2f}ms, success={success}")
    
    # Convert results to DataFrame
    return pd.DataFrame(results)

def visualize_results(results_df):
    """Visualize benchmark results."""
    plt.figure(figsize=(12, 10))
    
    # Plot average time by method
    plt.subplot(2, 1, 1)
    sns.barplot(x="method", y="time_ms", data=results_df, estimator=np.mean)
    plt.title("Average Embedding Time (ms)")
    plt.ylabel("Time (ms)")
    plt.xlabel("Method")
    
    # Plot dimensions by method
    plt.subplot(2, 1, 2)
    sns.barplot(x="method", y="dimension", data=results_df)
    plt.title("Embedding Dimensions")
    plt.ylabel("Dimensions")
    plt.xlabel("Method")
    
    plt.tight_layout()
    plt.savefig("embedding_benchmark_results.png")
    plt.close()
    
    logger.info("Created visualization: embedding_benchmark_results.png")

def main():
    """Run embedding benchmark."""
    # Test texts
    texts = [
        "Java developer with 5 years experience in Spring Boot and React",
        "Data scientist skilled in Python, machine learning, and big data technologies",
        "Project manager with agile certification and experience in software development",
        "UX designer with portfolio of mobile and web app designs"
    ]
    
    # Run benchmark
    results = benchmark_embeddings(texts)
    
    # Print summary
    print("\nEmbedding Benchmark Summary:")
    print("-" * 50)
    
    summary = results.groupby("method").agg({
        "time_ms": ["mean", "std", "min", "max"],
        "dimension": "first",
        "success": "mean"
    })
    
    print(summary)
    
    # Visualize results
    visualize_results(results)
    
    # Save detailed results
    results.to_csv("embedding_benchmark_results.csv", index=False)
    logger.info("Saved detailed results to embedding_benchmark_results.csv")

if __name__ == "__main__":
    main() 