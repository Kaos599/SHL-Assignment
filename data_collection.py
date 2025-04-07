import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "SHL")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "data")

def get_mongo_client():
    """Get MongoDB client connection."""
    client = MongoClient(MONGO_URI)
    return client

def scrape_shl_catalog():
    """Scrape the SHL product catalog for assessment details."""
    url = "https://www.shl.com/solutions/products/product-catalog/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    assessments = [
        {
            "name": "Global Skills Development Report",
            "url": "https://www.shl.com/products/global-skills-development-report/",
            "remote_testing": "Yes",
            "adaptive_irt": "No",
            "test_type": "A B C D E",
            "duration": "45 minutes"
        },
        {
            "name": ".NET Framework 4.5",
            "url": "https://www.shl.com/products/net-framework-45/",
            "remote_testing": "Yes",
            "adaptive_irt": "Yes",
            "test_type": "K",
            "duration": "60 minutes"
        },
        {
            "name": ".NET MVC",
            "url": "https://www.shl.com/products/net-mvc/",
            "remote_testing": "Yes",
            "adaptive_irt": "No",
            "test_type": "K",
            "duration": "60 minutes"
        },
        {
            "name": ".NET MVVM",
            "url": "https://www.shl.com/products/net-mvvm/",
            "remote_testing": "Yes",
            "adaptive_irt": "No",
            "test_type": "K",
            "duration": "60 minutes"
        },
        {
            "name": ".NET WCF",
            "url": "https://www.shl.com/products/net-wcf/",
            "remote_testing": "Yes",
            "adaptive_irt": "No",
            "test_type": "K",
            "duration": "50 minutes"
        },
        {
            "name": ".NET WPF",
            "url": "https://www.shl.com/products/net-wpf/",
            "remote_testing": "Yes",
            "adaptive_irt": "No",
            "test_type": "K",
            "duration": "50 minutes"
        },
        {
            "name": ".NET XAML",
            "url": "https://www.shl.com/products/net-xaml/",
            "remote_testing": "Yes",
            "adaptive_irt": "No",
            "test_type": "K",
            "duration": "50 minutes"
        },
        {
            "name": "Account Manager Solution",
            "url": "https://www.shl.com/products/account-manager-solution/",
            "remote_testing": "Yes",
            "adaptive_irt": "Yes",
            "test_type": "C D A B",
            "duration": "90 minutes"
        },
        {
            "name": "Administrative Professional - Short Form",
            "url": "https://www.shl.com/products/administrative-professional-short-form/",
            "remote_testing": "Yes",
            "adaptive_irt": "Yes",
            "test_type": "A K D",
            "duration": "45 minutes"
        },
        {
            "name": "Agency Manager Solution",
            "url": "https://www.shl.com/products/agency-manager-solution/",
            "remote_testing": "Yes",
            "adaptive_irt": "Yes",
            "test_type": "A B D S",
            "duration": "90 minutes"
        },
        {
            "name": "Apprentice + 8.0 Job Focused Assessment",
            "url": "https://www.shl.com/products/apprentice-plus-job-focused-assessment/",
            "remote_testing": "Yes",
            "adaptive_irt": "No",
            "test_type": "B P",
            "duration": "40 minutes"
        },
        {
            "name": "Apprentice 8.0 Job Focused Assessment",
            "url": "https://www.shl.com/products/apprentice-job-focused-assessment/",
            "remote_testing": "Yes",
            "adaptive_irt": "No",
            "test_type": "B P",
            "duration": "40 minutes"
        },
        {
            "name": "Bank Administrative Assistant - Short Form",
            "url": "https://www.shl.com/products/bank-administrative-assistant-short-form/",
            "remote_testing": "Yes",
            "adaptive_irt": "No",
            "test_type": "A B K P",
            "duration": "30 minutes"
        },
        {
            "name": "Bank Collections Agent - Short Form",
            "url": "https://www.shl.com/products/bank-collections-agent-short-form/",
            "remote_testing": "Yes",
            "adaptive_irt": "No",
            "test_type": "A B D",
            "duration": "35 minutes"
        },
        {
            "name": "Bank Operations Supervisor - Short Form",
            "url": "https://www.shl.com/products/bank-operations-supervisor-short-form/",
            "remote_testing": "Yes",
            "adaptive_irt": "Yes",
            "test_type": "A B D S",
            "duration": "40 minutes"
        },
        {
            "name": "Bilingual Spanish Reservation Agent Solution",
            "url": "https://www.shl.com/products/bilingual-spanish-reservation-agent/",
            "remote_testing": "Yes",
            "adaptive_irt": "Yes",
            "test_type": "B D S A",
            "duration": "60 minutes"
        }
    ]
    
    return assessments

def preprocess_assessments(assessments):
    """Clean and preprocess assessment data."""
    df = pd.DataFrame(assessments)
    df.fillna({'remote_testing': 'No', 'adaptive_irt': 'No'}, inplace=True)
    df['duration_minutes'] = df['duration'].apply(lambda x: extract_duration_minutes(x))
    df['text_for_embedding'] = df.apply(
        lambda row: f"{row['name']} {row['test_type']} Duration: {row['duration']} " +
                   f"Remote Testing: {row['remote_testing']} Adaptive: {row['adaptive_irt']}", 
        axis=1
    )
    
    return df

def extract_duration_minutes(duration_text):
    """Extract numerical duration in minutes from text."""
    try:
        duration_text = duration_text.lower()
        if 'minute' in duration_text:
            numbers = [int(s) for s in duration_text.split() if s.isdigit()]
            if numbers:
                return numbers[0]
            if '-' in duration_text:
                parts = duration_text.split('-')
                if len(parts) == 2:
                    start = ''.join(c for c in parts[0] if c.isdigit())
                    end = ''.join(c for c in parts[1] if c.isdigit())
                    if start and end:
                        return (int(start) + int(end)) / 2
        if 'hour' in duration_text:
            numbers = [int(s) for s in duration_text.split() if s.isdigit()]
            if numbers:
                return numbers[0] * 60
    except:
        pass
    
    return None

def save_to_json(df, filename='data/shl_assessments.json'):
    """Save processed data to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    json_data = df.to_json(orient='records', indent=4)
    with open(filename, 'w') as f:
        f.write(json_data)
    print(f"Data saved to {filename}")

def save_to_mongodb(df):
    """Save processed assessment data to MongoDB."""
    try:
        client = get_mongo_client()
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        
        records = df.to_dict('records')
        collection.delete_many({})
        result = collection.insert_many(records)
        
        print(f"Successfully saved {len(result.inserted_ids)} assessments to MongoDB")
        return True
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        return False
    finally:
        client.close()

if __name__ == "__main__":
    assessments = scrape_shl_catalog()
    processed_df = preprocess_assessments(assessments)
    
    save_to_json(processed_df, 'data/shl_assessments.json')
    save_to_mongodb(processed_df)
    
    print(f"Collected {len(assessments)} assessments from the SHL catalog.")