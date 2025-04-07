import os
import json
import datetime
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configure Firecrawl and MongoDB connections
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "SHL")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "data")

# MongoDB connection flag
USE_MONGODB = True

def get_mongo_client():
    """Get MongoDB client connection."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')  # Test connection
        print("MongoDB connection successful")
        return client
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        global USE_MONGODB
        USE_MONGODB = False
        return None

def scrape_shl_with_firecrawl():
    """Scrape SHL website using Firecrawl and store structured data in MongoDB."""
    print("Initializing Firecrawl...")
    firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    
    product_catalog_url = "https://www.shl.com/solutions/products/product-catalog/"
    
    # Define the schema for structured data extraction
    extraction_schema = {
        "type": "object",
        "properties": {
            "products": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "test_type": {"type": "string"},
                        "duration": {"type": "string"},
                        "remote_testing": {"type": "string"},
                        "adaptive_irt": {"type": "string"},
                        "url": {"type": "string"}
                    }
                },
                "description": "List of SHL assessment products"
            }
        }
    }
    
    try:
        print(f"Scraping SHL product catalog: {product_catalog_url}")
        
        result = firecrawl.scrape_url(
            url=product_catalog_url,
            params={
                "formats": ["json", "markdown"],
                "jsonOptions": {
                    "schema": extraction_schema
                }
            }
        )
        
        print("Raw result keys:", result.keys())
        
        if "json" in result:
            structured_data = result["json"]
            structured_data["source_url"] = product_catalog_url
            structured_data["scraped_at"] = datetime.datetime.now().isoformat()
            
            # Save to local JSON file
            os.makedirs("data", exist_ok=True)
            json_filepath = "data/shl_firecrawl_data.json"
            with open(json_filepath, "w") as f:
                json.dump(structured_data, f, indent=2)
            print(f"Saved data to {json_filepath}")
            
            # Try to save to MongoDB if enabled
            if USE_MONGODB:
                try:
                    client = get_mongo_client()
                    if client:
                        db = client[MONGO_DB]
                        collection = db[MONGO_COLLECTION]
                        insert_result = collection.insert_one(structured_data)
                        print(f"Saved structured data to MongoDB with ID: {insert_result.inserted_id}")
                    else:
                        print("Skipping MongoDB storage due to connection issues")
                except Exception as mongo_ex:
                    print(f"Error saving to MongoDB: {mongo_ex}")
                    print("Data was saved locally only")
            else:
                print("MongoDB storage is disabled. Data saved locally only.")
            
            return structured_data
        else:
            print("No structured data extracted. Raw result:", result)
            return None
            
    except Exception as e:
        print(f"Error during scraping or storing data: {e}")
        return None

def crawl_shl_site():
    """Crawl multiple pages of the SHL website and store in MongoDB."""
    print("Initializing Firecrawl for full site crawl...")
    firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    
    base_url = "https://www.shl.com"
    
    try:
        print(f"Crawling SHL website: {base_url}")
        
        result = firecrawl.crawl_url(
            url=base_url,
            params={
                "excludePaths": ["/blog/*", "/resources/*"],
                "maxDepth": 2,
                "limit": 50,
                "scrapeOptions": {
                    "formats": ["markdown"],
                    "onlyMainContent": True
                }
            }
        )
        
        print("Crawl result keys:", result.keys())
        
        if "pages" in result and isinstance(result["pages"], list):
            pages = result["pages"]
            print(f"Crawled {len(pages)} pages from SHL website")
            
            # Prepare data for MongoDB
            processed_pages = []
            for page in pages:
                page_data = {
                    "url": page.get("url"),
                    "title": page.get("title"),
                    "content": page.get("markdown"),
                    "crawl_time": datetime.datetime.now().isoformat()
                }
                processed_pages.append(page_data)
            
            # Save to local JSON file
            os.makedirs("data", exist_ok=True)
            json_filepath = "data/shl_crawled_pages.json"
            with open(json_filepath, "w") as f:
                json.dump(processed_pages, f, indent=2)
            print(f"Saved {len(processed_pages)} crawled pages to {json_filepath}")
            
            # Save first page as example
            with open("data/shl_sample_page.json", "w") as f:
                json.dump(processed_pages[0] if processed_pages else {}, f, indent=2)
            
            # Try to save to MongoDB if enabled
            if USE_MONGODB and processed_pages:
                try:
                    client = get_mongo_client()
                    if client:
                        db = client[MONGO_DB]
                        collection = db[MONGO_COLLECTION + "_pages"]  # Use a different collection
                        insert_result = collection.insert_many(processed_pages)
                        print(f"Saved {len(insert_result.inserted_ids)} pages to MongoDB")
                    else:
                        print("Skipping MongoDB storage due to connection issues")
                except Exception as mongo_ex:
                    print(f"Error saving to MongoDB: {mongo_ex}")
                    print("Data was saved locally only")
            else:
                if not processed_pages:
                    print("No pages to save")
                else:
                    print("MongoDB storage is disabled. Data saved locally only.")
            
            return len(processed_pages)
        else:
            print("No pages found in crawl result. Full result:", result)
            return 0
            
    except Exception as e:
        print(f"Error during crawling: {e}")
        return None

if __name__ == "__main__":
    # Test MongoDB connection at startup
    mongo_client = get_mongo_client()
    if not mongo_client:
        print("Will proceed without MongoDB storage. Data will be saved locally only.")
    
    # Automatically run the scrape function without asking for input
    print("Starting SHL Website Scraper using Firecrawl")
    scrape_shl_with_firecrawl() 