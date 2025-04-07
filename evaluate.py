import pandas as pd
import numpy as np
import json
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Set evaluation parameters
API_URL = "http://localhost:8000/recommend"
K_VALUES = [1, 3, 5, 10]

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "SHL")
MONGO_COLLECTION_PACKAGED = os.getenv("MONGO_COLLECTION_PACKAGED", "packaged_solutions")
MONGO_COLLECTION_INDIVIDUAL = os.getenv("MONGO_COLLECTION_INDIVIDUAL", "individual_solutions")

def get_mongo_client():
    """Get MongoDB client connection."""
    client = MongoClient(MONGO_URI)
    return client

def get_real_assessment_names():
    """Get real assessment names from MongoDB to use in evaluation."""
    try:
        client = get_mongo_client()
        db = client[MONGO_DB]
        
        # Get assessment names from packaged solutions
        packaged_collection = db[MONGO_COLLECTION_PACKAGED]
        packaged_assessments = list(packaged_collection.find({}, {"name": 1, "test_type_names": 1, "job_levels": 1}))
        
        # Get assessment names from individual solutions
        individual_collection = db[MONGO_COLLECTION_INDIVIDUAL]
        individual_assessments = list(individual_collection.find({}, {"name": 1, "test_type_names": 1, "job_levels": 1}))
        
        return packaged_assessments + individual_assessments
    except Exception as e:
        print(f"Error loading from MongoDB: {e}")
        return []
    finally:
        client.close()

def load_test_queries():
    """Load test queries and their ground truth relevant assessments."""
    # Get real assessment names from MongoDB
    real_assessments = get_real_assessment_names()
    
    # If no real assessments found, use placeholder data
    if not real_assessments:
        print("Warning: No real assessments found in MongoDB. Using placeholder data.")
        return [
            {
                "id": "q1",
                "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
                "relevant_assessments": ["SHL Java Programming Test", "SHL Collaboration Assessment", "SHL Agile Development Assessment"]
            },
            {
                "id": "q2",
                "query": "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",
                "relevant_assessments": ["SHL Python Test", "SHL SQL Assessment", "SHL JavaScript Evaluation"]
            },
            {
                "id": "q3",
                "query": "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins.",
                "relevant_assessments": ["SHL Cognitive Assessment", "SHL Personality Profile", "SHL Analyst Skills Test"]
            }
        ]
    
    # Find assessments for Java developers with collaborative skills
    java_assessments = [a["name"] for a in real_assessments 
                        if any(test_type in ["Competencies", "Knowledge & Skills"] 
                            for test_type in a.get("test_type_names", []))][:3]
    
    # Find assessments for technical skills (Python, SQL, JavaScript)
    tech_assessments = [a["name"] for a in real_assessments 
                        if any(test_type in ["Knowledge & Skills", "Ability & Aptitude"] 
                            for test_type in a.get("test_type_names", []))][:3]
    
    # Find assessments for analysts (cognitive and personality)
    analyst_assessments = [a["name"] for a in real_assessments 
                          if any(test_type in ["Personality & Behavior", "Ability & Aptitude"] 
                             for test_type in a.get("test_type_names", []))][:3]
    
    # Create test queries with real assessment names
    test_data = [
        {
            "id": "q1",
            "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
            "relevant_assessments": java_assessments if java_assessments else ["Unknown"]
        },
        {
            "id": "q2",
            "query": "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript. Need an assessment package that can test all skills with max duration of 60 minutes.",
            "relevant_assessments": tech_assessments if tech_assessments else ["Unknown"]
        },
        {
            "id": "q3",
            "query": "I am hiring for an analyst and want to screen applications using Cognitive and personality tests. What options are available within 45 minutes?",
            "relevant_assessments": analyst_assessments if analyst_assessments else ["Unknown"]
        },
        {
            "id": "q4", 
            "query": "We need to hire customer service representatives. Looking for a package that assesses interpersonal skills and problem-solving ability.",
            "relevant_assessments": [a["name"] for a in real_assessments 
                                    if "Customer Service" in a.get("name", "")][:3]
        },
        {
            "id": "q5",
            "query": "Need to assess leadership potential for executive candidates. Prefer assessments that are adaptive and can be taken remotely.",
            "relevant_assessments": [a["name"] for a in real_assessments 
                                    if "Leadership" in a.get("name", "") or
                                    any("Leadership" in test_type for test_type in a.get("test_type_names", []))][:3]
        }
    ]
    
    # Print the test queries and their relevant assessments
    print("Test Queries and Relevant Assessments:")
    for query in test_data:
        print(f"Query {query['id']}: {query['query']}")
        print(f"Relevant Assessments: {query['relevant_assessments']}")
        print()
    
    return test_data

def calculate_recall_at_k(recommendations, relevant_items, k):
    """Calculate Recall@K metric."""
    if not relevant_items or relevant_items == ["Unknown"]:
        return 0.0
    
    top_k_recs = recommendations[:k]
    relevant_in_top_k = sum(1 for rec in top_k_recs if rec['name'] in relevant_items)
    
    return relevant_in_top_k / len(relevant_items)

def calculate_ap_at_k(recommendations, relevant_items, k):
    """Calculate Average Precision@K."""
    if not relevant_items or relevant_items == ["Unknown"]:
        return 0.0
    
    top_k_recs = recommendations[:k]
    relevant_count = 0
    sum_precisions = 0.0
    
    for i, rec in enumerate(top_k_recs):
        if rec['name'] in relevant_items:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            sum_precisions += precision_at_i
    
    if min(k, len(relevant_items)) == 0:
        return 0.0
    
    return sum_precisions / min(k, len(relevant_items))

def evaluate_model():
    """Evaluate the recommendation model on test queries."""
    test_queries = load_test_queries()
    
    results = {
        'query_id': [],
        'query': [],
        'recall@1': [],
        'recall@3': [],
        'recall@5': [],
        'recall@10': [],
        'ap@1': [],
        'ap@3': [],
        'ap@5': [],
        'ap@10': []
    }
    
    for query_data in tqdm(test_queries, desc="Evaluating queries"):
        query_id = query_data['id']
        query = query_data['query']
        relevant_assessments = query_data['relevant_assessments']
        
        try:
            response = requests.get(API_URL, params={"query": query, "max_results": max(K_VALUES)})
            
            if response.status_code == 200:
                data = response.json()
                recommendations = data['recommendations']
                
                results['query_id'].append(query_id)
                results['query'].append(query)
                
                for k in K_VALUES:
                    recall = calculate_recall_at_k(recommendations, relevant_assessments, k)
                    ap = calculate_ap_at_k(recommendations, relevant_assessments, k)
                    
                    results[f'recall@{k}'].append(recall)
                    results[f'ap@{k}'].append(ap)
            else:
                print(f"Error for query {query_id}: {response.status_code} - {response.text}")
                # Add empty results for this query
                results['query_id'].append(query_id)
                results['query'].append(query)
                for k in K_VALUES:
                    results[f'recall@{k}'].append(0.0)
                    results[f'ap@{k}'].append(0.0)
        except Exception as e:
            print(f"Exception for query {query_id}: {e}")
            # Add empty results for this query
            results['query_id'].append(query_id)
            results['query'].append(query)
            for k in K_VALUES:
                results[f'recall@{k}'].append(0.0)
                results[f'ap@{k}'].append(0.0)
    
    results_df = pd.DataFrame(results)
    
    mean_metrics = {
        'mean_recall@1': results_df['recall@1'].mean(),
        'mean_recall@3': results_df['recall@3'].mean(),
        'mean_recall@5': results_df['recall@5'].mean(),
        'mean_recall@10': results_df['recall@10'].mean(),
        'mean_ap@1': results_df['ap@1'].mean(),
        'mean_ap@3': results_df['ap@3'].mean(),
        'mean_ap@5': results_df['ap@5'].mean(),
        'mean_ap@10': results_df['ap@10'].mean()
    }
    
    return results_df, mean_metrics

def plot_results(results_df, mean_metrics):
    """Plot evaluation results."""
    plt.figure(figsize=(15, 10))
    
    recall_values = [mean_metrics[f'mean_recall@{k}'] for k in K_VALUES]
    ap_values = [mean_metrics[f'mean_ap@{k}'] for k in K_VALUES]
    
    # Plot Mean Recall@K
    plt.subplot(2, 2, 1)
    plt.plot(K_VALUES, recall_values, 'o-', label='Mean Recall@K')
    plt.xlabel('K')
    plt.ylabel('Mean Recall@K')
    plt.title('Mean Recall@K for Different K Values')
    plt.grid(True)
    
    # Plot Mean AP@K
    plt.subplot(2, 2, 2)
    plt.plot(K_VALUES, ap_values, 'o-', label='Mean AP@K')
    plt.xlabel('K')
    plt.ylabel('Mean AP@K')
    plt.title('Mean Average Precision@K for Different K Values')
    plt.grid(True)
    
    # Plot Recall@K for each query
    plt.subplot(2, 2, 3)
    for idx, row in results_df.iterrows():
        query_id = row['query_id']
        recalls = [row[f'recall@{k}'] for k in K_VALUES]
        plt.plot(K_VALUES, recalls, 'o-', label=f'Query {query_id}')
    plt.xlabel('K')
    plt.ylabel('Recall@K')
    plt.title('Recall@K by Query')
    plt.grid(True)
    plt.legend()
    
    # Plot AP@K for each query
    plt.subplot(2, 2, 4)
    for idx, row in results_df.iterrows():
        query_id = row['query_id']
        aps = [row[f'ap@{k}'] for k in K_VALUES]
        plt.plot(K_VALUES, aps, 'o-', label=f'Query {query_id}')
    plt.xlabel('K')
    plt.ylabel('AP@K')
    plt.title('Average Precision@K by Query')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    plt.savefig('data/evaluation_results.png')
    plt.close()
    
    with open('data/evaluation_metrics.json', 'w') as f:
        json.dump(mean_metrics, f, indent=4)
    
    # Save the detailed results
    results_df.to_csv('data/evaluation_results.csv', index=False)
    
    print("Evaluation results saved to data/evaluation_results.png")
    print("Evaluation metrics saved to data/evaluation_metrics.json")
    print("Detailed results saved to data/evaluation_results.csv")

if __name__ == "__main__":
    print("Starting evaluation...")
    results_df, mean_metrics = evaluate_model()
    
    print("\nEvaluation Results:")
    for k in K_VALUES:
        print(f"Mean Recall@{k}: {mean_metrics[f'mean_recall@{k}']:.4f}")
        print(f"Mean AP@{k}: {mean_metrics[f'mean_ap@{k}']:.4f}")
    
    plot_results(results_df, mean_metrics)
    
    print("\nDetailed results by query:")
    print(results_df)