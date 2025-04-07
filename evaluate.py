import pandas as pd
import numpy as np
import json
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set evaluation parameters
API_URL = "http://localhost:8000/recommend"
K_VALUES = [1, 3, 5, 10]

def load_test_queries():
    """Load test queries and their ground truth relevant assessments."""
    test_data = [
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
    
    return test_data

def calculate_recall_at_k(recommendations, relevant_items, k):
    """Calculate Recall@K metric."""
    if not relevant_items:
        return 0.0
    
    top_k_recs = recommendations[:k]
    relevant_in_top_k = sum(1 for rec in top_k_recs if rec['name'] in relevant_items)
    
    return relevant_in_top_k / len(relevant_items)

def calculate_ap_at_k(recommendations, relevant_items, k):
    """Calculate Average Precision@K."""
    if not relevant_items:
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
    plt.figure(figsize=(12, 8))
    
    recall_values = [mean_metrics[f'mean_recall@{k}'] for k in K_VALUES]
    ap_values = [mean_metrics[f'mean_ap@{k}'] for k in K_VALUES]
    
    plt.subplot(2, 1, 1)
    plt.plot(K_VALUES, recall_values, 'o-', label='Mean Recall@K')
    plt.xlabel('K')
    plt.ylabel('Mean Recall@K')
    plt.title('Mean Recall@K for Different K Values')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(K_VALUES, ap_values, 'o-', label='Mean AP@K')
    plt.xlabel('K')
    plt.ylabel('Mean AP@K')
    plt.title('Mean Average Precision@K for Different K Values')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.close()
    
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(mean_metrics, f, indent=4)
    
    print("Evaluation results saved to evaluation_results.png")
    print("Evaluation metrics saved to evaluation_metrics.json")

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