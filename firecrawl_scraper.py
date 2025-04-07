import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient
import time
import asyncio
import nest_asyncio
from firecrawl import FirecrawlApp

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "SHL")
MONGO_COLLECTION_PACKAGED = os.getenv("MONGO_COLLECTION_PACKAGED", "packaged_solutions")
MONGO_COLLECTION_INDIVIDUAL = os.getenv("MONGO_COLLECTION_INDIVIDUAL", "individual_solutions")

# Firecrawl API key
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# SHL catalog URLs
BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
PACKAGED_SOLUTIONS_PAGES = [f"{BASE_URL}?start={i*12}&type=2" for i in range(13)]  # Going until page 12 as mentioned
INDIVIDUAL_SOLUTIONS_PAGES = [f"{BASE_URL}?start={i*12}&type=1" for i in range(32)]  # Going until start=372 in increments of 12

def get_mongo_client():
    """Get MongoDB client connection."""
    try:
        client = MongoClient(MONGO_URI)
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def save_to_mongodb(data, collection_name):
    """Save processed assessment data to MongoDB."""
    try:
        client = get_mongo_client()
        db = client[MONGO_DB]
        collection = db[collection_name]
        
        if not data:
            print(f"No data to save to {collection_name}")
            return False
            
        # Upsert based on name to avoid duplicates
        for item in data:
            if "name" in item:
                collection.update_one(
                    {"name": item["name"]},
                    {"$set": item},
                    upsert=True
                )
            else:
                collection.insert_one(item)
        
        print(f"Successfully saved {len(data)} assessments to MongoDB collection {collection_name}")
        return True
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        return False
    finally:
        if client:
            client.close()

def save_to_json(data, filename):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {filename}")

def parse_solution_details(data):
    """Parse structured data from a solution detail page."""
    solutions = []
    
    if not data or not isinstance(data, list):
        return solutions
    
    for page in data:
        if not isinstance(page, dict) or 'markdown' not in page:
            continue
            
        markdown = page['markdown']
        metadata = page.get('metadata', {})
        url = metadata.get('sourceURL', '')
        
        # Skip if not a product detail page
        if not url or '/products/' not in url:
            continue
            
        # Basic solution info
        solution = {
            "name": metadata.get('title', '').replace(' | SHL', ''),
            "url": url,
            "description": "",
            "job_levels": [],
            "languages": [],
            "assessment_length": "",
            "remote_testing": False,
            "adaptive_irt": False,
            "test_type": [],
            "test_type_details": {}  # Store detailed test type information
        }
        
        # Parse the markdown content
        lines = markdown.split('\n')
        section = None
        current_test_type = None
        
        for line in lines:
            line = line.strip()
            
            # Extract description
            if line.startswith('Description'):
                section = 'description'
                continue
            elif line.startswith('Job levels'):
                section = 'job_levels'
                continue
            elif line.startswith('Languages'):
                section = 'languages'
                continue
            elif line.startswith('Assessment length'):
                section = 'assessment_length'
                continue
            elif line.startswith('Test Type'):
                section = 'test_type'
                continue
            elif line.startswith('Remote Testing'):
                section = 'remote_testing'
                continue
            elif any(test_type in line for test_type in ['A', 'B', 'C', 'D', 'P', 'K', 'S']):
                # Found a test type line
                test_type = line.split()[0]  # Get the test type letter
                solution['test_type'].append(test_type)
                current_test_type = test_type
                solution['test_type_details'][test_type] = {
                    'name': line.split('-')[1].strip() if '-' in line else line.strip(),
                    'description': ''
                }
                continue
            
            # Collect data for the current section
            if section == 'description' and line and not line.startswith('#'):
                solution['description'] += line + " "
            elif section == 'job_levels' and line and not line.startswith('#'):
                solution['job_levels'].extend([l.strip() for l in line.split(',')])
            elif section == 'languages' and line and not line.startswith('#'):
                solution['languages'].extend([l.strip() for l in line.split(',')])
            elif section == 'assessment_length' and line and not line.startswith('#'):
                solution['assessment_length'] = line
            elif section == 'remote_testing':
                solution['remote_testing'] = 'yes' in line.lower() or 'true' in line.lower()
                section = None
            elif current_test_type and line and not line.startswith('#'):
                # Add description for current test type
                solution['test_type_details'][current_test_type]['description'] += line + " "
        
        solution['description'] = solution['description'].strip()
        
        # Check for Adaptive/IRT in the description or any section
        solution['adaptive_irt'] = 'adaptive' in markdown.lower() or 'irt' in markdown.lower()
        
        # Clean up the job levels and languages
        solution['job_levels'] = [jl for jl in solution['job_levels'] if jl]
        solution['languages'] = [l for l in solution['languages'] if l]
        
        # Clean up test type details
        for test_type in solution['test_type_details']:
            solution['test_type_details'][test_type]['description'] = solution['test_type_details'][test_type]['description'].strip()
        
        # Add to solutions list
        if solution['name']:
            solutions.append(solution)
    
    return solutions

def parse_catalog_page(data, solution_type):
    """Parse the catalog page to extract solutions and their properties."""
    solutions = []
    
    if not data or not isinstance(data, list):
        return solutions
    
    for page in data:
        if not isinstance(page, dict) or 'markdown' not in page:
            continue
            
        markdown = page['markdown']
        
        # Look for the table with solutions
        if 'Pre-packaged Job Solutions' in markdown and solution_type == 'packaged':
            # Process table
            solution_table_rows = markdown.split('Pre-packaged Job Solutions')[1].split('\n')
            current_solution = {}
            
            for row in solution_table_rows:
                # If we find a link, it's a new solution
                if '[' in row and '](' in row:
                    # Save previous solution if it exists
                    if current_solution and 'name' in current_solution:
                        solutions.append(current_solution)
                    
                    # Start new solution
                    name_part = row.split('](')[0][1:]
                    url_part = row.split('](')[1].split(')')[0]
                    current_solution = {
                        "name": name_part,
                        "url": url_part,
                        "remote_testing": False,
                        "adaptive_irt": False,
                        "test_type": [],
                        "test_type_details": {}
                    }
                
                # Check for test properties
                if current_solution:
                    if '●' in row:  # Remote Testing or Adaptive/IRT indicator
                        if 'Remote Testing' in row:
                            current_solution['remote_testing'] = True
                        if 'Adaptive/IRT' in row:
                            current_solution['adaptive_irt'] = True
                    
                    # Extract test types (A B C D P, etc.)
                    test_types = [t for t in row.split() if len(t) == 1 and t.isalpha() and t in ['A', 'B', 'C', 'D', 'P', 'K', 'S']]
                    for test_type in test_types:
                        if test_type not in current_solution['test_type']:
                            current_solution['test_type'].append(test_type)
                            current_solution['test_type_details'][test_type] = {
                                'name': '',  # Will be filled in from detail page
                                'description': ''  # Will be filled in from detail page
                            }
            
            # Add the last solution if it exists
            if current_solution and 'name' in current_solution:
                solutions.append(current_solution)
        
        elif 'Individual Test Solutions' in markdown and solution_type == 'individual':
            # Similar processing for individual test solutions
            solution_table_rows = markdown.split('Individual Test Solutions')[1].split('\n')
            current_solution = {}
            
            for row in solution_table_rows:
                # If we find a link, it's a new solution
                if '[' in row and '](' in row:
                    # Save previous solution if it exists
                    if current_solution and 'name' in current_solution:
                        solutions.append(current_solution)
                    
                    # Start new solution
                    name_part = row.split('](')[0][1:]
                    url_part = row.split('](')[1].split(')')[0]
                    current_solution = {
                        "name": name_part,
                        "url": url_part,
                        "remote_testing": False,
                        "adaptive_irt": False,
                        "test_type": [],
                        "test_type_details": {}
                    }
                
                # Check for test properties
                if current_solution:
                    if '●' in row:  # Remote Testing or Adaptive/IRT indicator
                        if 'Remote Testing' in row:
                            current_solution['remote_testing'] = True
                        if 'Adaptive/IRT' in row:
                            current_solution['adaptive_irt'] = True
                    
                    # Extract test types (A B C D P, etc.)
                    test_types = [t for t in row.split() if len(t) == 1 and t.isalpha() and t in ['A', 'B', 'C', 'D', 'P', 'K', 'S']]
                    for test_type in test_types:
                        if test_type not in current_solution['test_type']:
                            current_solution['test_type'].append(test_type)
                            current_solution['test_type_details'][test_type] = {
                                'name': '',  # Will be filled in from detail page
                                'description': ''  # Will be filled in from detail page
                            }
            
            # Add the last solution if it exists
            if current_solution and 'name' in current_solution:
                solutions.append(current_solution)
    
    return solutions

async def scrape_shl_catalog():
    """Scrape the SHL product catalog using Firecrawl."""
    if not FIRECRAWL_API_KEY:
        print("Error: FIRECRAWL_API_KEY environment variable not set")
        return
    
    # Initialize Firecrawl
    app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    
    # Apply nest_asyncio for Jupyter/interactive environments
    nest_asyncio.apply()
    
    async def scrape_page(url, solution_type):
        """Scrape a single page and its solutions."""
        try:
            print(f"Crawling {url}...")
            # Crawl the catalog page with optimized parameters
            catalog_data = await app.crawl_url(
                url,
                params={
                    'limit': 10,  # Limit to just the catalog page
                    'scrapeOptions': {
                        'formats': ['markdown'],
                        'onlyMainContent': True,  # Only get main content
                        'includeHtml': False,  # Don't include raw HTML
                        'waitFor': 2000  # Wait 2 seconds for dynamic content
                    }
                },
                poll_interval=2  # Reduced poll interval
            )
            
            # Parse the catalog page
            page_solutions = parse_catalog_page(catalog_data, solution_type)
            
            # Concurrently scrape detail pages
            detail_tasks = []
            for solution in page_solutions:
                if 'url' in solution and solution['url']:
                    detail_tasks.append(scrape_solution_detail(app, solution))
            
            # Wait for all detail pages to be scraped
            if detail_tasks:
                await asyncio.gather(*detail_tasks)
            
            return page_solutions
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return []

    async def scrape_solution_detail(app, solution):
        """Scrape a single solution's detail page."""
        try:
            print(f"Crawling detail page: {solution['url']}...")
            detail_data = await app.crawl_url(
                solution['url'],
                params={
                    'limit': 1,
                    'scrapeOptions': {
                        'formats': ['markdown'],
                        'onlyMainContent': True,
                        'includeHtml': False,
                        'waitFor': 2000
                    }
                },
                poll_interval=2
            )
            
            # Parse the detail page and merge with catalog data
            detail_solutions = parse_solution_details(detail_data)
            if detail_solutions:
                # Merge the detail data with the existing solution
                for field, value in detail_solutions[0].items():
                    if field not in solution or not solution[field]:
                        solution[field] = value
        except Exception as e:
            print(f"Error crawling detail page {solution['url']}: {e}")

    # Scrape packaged solutions concurrently
    packaged_tasks = [scrape_page(url, 'packaged') for url in PACKAGED_SOLUTIONS_PAGES]
    packaged_solutions = await asyncio.gather(*packaged_tasks)
    packaged_solutions = [sol for page in packaged_solutions for sol in page]  # Flatten list
    
    # Save packaged solutions
    save_to_mongodb(packaged_solutions, MONGO_COLLECTION_PACKAGED)
    save_to_json(packaged_solutions, 'data/packaged_solutions.json')
    
    # Scrape individual solutions concurrently
    individual_tasks = [scrape_page(url, 'individual') for url in INDIVIDUAL_SOLUTIONS_PAGES]
    individual_solutions = await asyncio.gather(*individual_tasks)
    individual_solutions = [sol for page in individual_solutions for sol in page]  # Flatten list
    
    # Save individual solutions
    save_to_mongodb(individual_solutions, MONGO_COLLECTION_INDIVIDUAL)
    save_to_json(individual_solutions, 'data/individual_solutions.json')
    
    print("Scraping complete!")
    print(f"Collected {len(packaged_solutions)} packaged solutions and {len(individual_solutions)} individual solutions")
    
    return {
        'packaged': packaged_solutions,
        'individual': individual_solutions
    }

if __name__ == "__main__":
    # Use the modern asyncio.run() instead of deprecated get_event_loop()
    asyncio.run(scrape_shl_catalog())