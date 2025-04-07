import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
import faiss
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Azure OpenAI connection
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")

def load_test_data(sample_count=10):
    """Load or create test data for benchmarking."""
    # Sample test queries
    test_queries = [
        "Java developers who can collaborate with business teams",
        "Python, SQL and JavaScript programmers for mid-level roles",
        "Hiring analysts with cognitive and personality assessment needs",
        "Customer service representatives with interpersonal skills",
        "Leadership assessment for executive candidates",
        "Technical skills assessment for software engineers",
        "Sales team assessment with focus on negotiation skills",
        "Project managers with agile methodology expertise",
        "Data scientists with machine learning knowledge",
        "Financial analysts with attention to detail",
        "HR professionals with conflict resolution abilities",
        "Marketing specialists with creativity assessment",
        "DevOps engineers with automation skills",
        "Product managers with strategic thinking",
        "UX designers with user empathy"
    ]
    
    # Take the number of queries requested, or all if sample_count > len(test_queries)
    return test_queries[:min(sample_count, len(test_queries))]

def get_azure_openai_client():
    """Initialize and return Azure OpenAI client."""
    try:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        return client
    except Exception as e:
        logger.error(f"Error initializing Azure OpenAI client: {e}")
        return None

def get_azure_embedding(text, client):
    """Get embedding from Azure OpenAI API."""
    try:
        response = client.embeddings.create(
            input=text,
            model=AZURE_EMBEDDING_DEPLOYMENT_NAME
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting Azure embedding: {e}")
        return None

def get_sentence_transformer_embedding(text, model):
    """Get embedding from sentence-transformers."""
    try:
        embedding = model.encode([text])[0]
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error getting sentence-transformers embedding: {e}")
        return None

def run_benchmark(queries, repeat=3):
    """Run benchmarking tests for both embedding methods."""
    results = {
        "query": [],
        "method": [],
        "dimension": [],
        "time_ms": [],
        "success": []
    }
    
    # Initialize Azure OpenAI client
    azure_client = get_azure_openai_client()
    if not azure_client:
        logger.error("Azure OpenAI client initialization failed. Skipping Azure tests.")
    
    # Initialize sentence-transformers model
    try:
        st_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info(f"Loaded sentence-transformers model: all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Error loading sentence-transformers model: {e}")
        st_model = None
    
    # Run tests for each query
    for query in queries:
        logger.info(f"Testing query: {query[:50]}...")
        
        # Test Azure OpenAI embeddings
        if azure_client:
            for i in range(repeat):
                start_time = time.time()
                embedding = get_azure_embedding(query, azure_client)
                end_time = time.time()
                
                time_ms = (end_time - start_time) * 1000
                dimension = len(embedding) if embedding else 0
                success = embedding is not None
                
                results["query"].append(query)
                results["method"].append("Azure OpenAI")
                results["dimension"].append(dimension)
                results["time_ms"].append(time_ms)
                results["success"].append(success)
                
                logger.info(f"Azure OpenAI embedding: dim={dimension}, time={time_ms:.2f}ms, success={success}")
        
        # Test sentence-transformers embeddings
        if st_model:
            for i in range(repeat):
                start_time = time.time()
                embedding = get_sentence_transformer_embedding(query, st_model)
                end_time = time.time()
                
                time_ms = (end_time - start_time) * 1000
                dimension = len(embedding) if embedding else 0
                success = embedding is not None
                
                results["query"].append(query)
                results["method"].append("Sentence Transformers")
                results["dimension"].append(dimension)
                results["time_ms"].append(time_ms)
                results["success"].append(success)
                
                logger.info(f"Sentence Transformers embedding: dim={dimension}, time={time_ms:.2f}ms, success={success}")
    
    return pd.DataFrame(results)

def analyze_results(results_df):
    """Analyze and visualize benchmark results."""
    logger.info("Analyzing benchmark results...")
    
    # Calculate average metrics by method
    avg_metrics = results_df.groupby("method").agg({
        "time_ms": ["mean", "std", "min", "max"],
        "dimension": ["mean"],
        "success": ["mean"]
    }).reset_index()
    
    # Rename columns for better readability
    avg_metrics.columns = ["method", "avg_time_ms", "std_time_ms", "min_time_ms", "max_time_ms", "avg_dimension", "success_rate"]
    
    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"\n{avg_metrics.to_string(index=False)}")
    
    # Create comparison visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot average embedding time
    plt.subplot(2, 2, 1)
    sns.barplot(x="method", y="avg_time_ms", data=avg_metrics)
    plt.title("Average Embedding Time (ms)")
    plt.ylabel("Time (ms)")
    plt.xlabel("Method")
    
    # Plot embedding dimensions
    plt.subplot(2, 2, 2)
    sns.barplot(x="method", y="avg_dimension", data=avg_metrics)
    plt.title("Embedding Dimensions")
    plt.ylabel("Dimensions")
    plt.xlabel("Method")
    
    # Plot time distribution as boxplot
    plt.subplot(2, 2, 3)
    sns.boxplot(x="method", y="time_ms", data=results_df)
    plt.title("Embedding Time Distribution")
    plt.ylabel("Time (ms)")
    plt.xlabel("Method")
    
    # Plot success rate
    plt.subplot(2, 2, 4)
    sns.barplot(x="method", y="success_rate", data=avg_metrics)
    plt.title("Success Rate")
    plt.ylabel("Success Rate")
    plt.xlabel("Method")
    
    plt.tight_layout()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    plt.savefig('data/embedding_benchmark_results.png')
    logger.info("Saved visualization to data/embedding_benchmark_results.png")
    
    # Save results to CSV and JSON
    results_df.to_csv('data/embedding_benchmark_results.csv', index=False)
    with open('data/embedding_benchmark_summary.json', 'w') as f:
        json.dump(avg_metrics.to_dict(orient='records'), f, indent=4)
    
    logger.info("Saved detailed results to data/embedding_benchmark_results.csv")
    logger.info("Saved summary to data/embedding_benchmark_summary.json")
    
    return avg_metrics

def main():
    """Main function to run the benchmark."""
    logger.info("Starting embedding benchmark...")
    
    # Load test data
    queries = load_test_data(sample_count=10)
    logger.info(f"Loaded {len(queries)} test queries")
    
    # Run benchmark
    results = run_benchmark(queries, repeat=3)
    
    # Analyze results
    summary = analyze_results(results)
    
    logger.info("Benchmark completed!")

if __name__ == "__main__":
    main() 