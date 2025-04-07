import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
import time
from dotenv import load_dotenv
from pymongo import MongoClient
from urllib.parse import urljoin

# Import Selenium libraries for dynamic content
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "SHL")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "data")
MONGO_COLLECTION_PACKAGED = os.getenv("MONGO_COLLECTION_PACKAGED", "packaged_solutions")
MONGO_COLLECTION_INDIVIDUAL = os.getenv("MONGO_COLLECTION_INDIVIDUAL", "individual_solutions")

# Base URLs
BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
PACKAGED_SOLUTIONS_PAGES = [f"{BASE_URL}?start={i*12}&type=2" for i in range(13)]  # Going until page 12
INDIVIDUAL_SOLUTIONS_PAGES = [f"{BASE_URL}?start={i*12}&type=1" for i in range(32)]  # Going until start=372 in increments of 12

# Initialize WebDriver
def get_webdriver():
    """Initialize and return a configured Selenium WebDriver for Chrome."""
    options = Options()
    options.add_argument('--headless')  # Run in headless mode
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')  # Set a standard window size
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    
    try:
        driver = webdriver.Chrome(options=options)
        return driver
    except Exception as e:
        print(f"Error initializing WebDriver: {e}")
        return None

def get_mongo_client():
    """Get MongoDB client connection."""
    client = MongoClient(MONGO_URI)
    return client

def get_page_content(url, use_selenium=False):
    """Get page content using requests and BeautifulSoup or Selenium for dynamic content."""
    if not use_selenium:
        # Try with regular requests first
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url} with requests: {e}")
            # Fall back to Selenium if requests fails
            use_selenium = True
    
    if use_selenium:
        driver = get_webdriver()
        if not driver:
            return None
        
        try:
            print(f"Using Selenium to fetch {url}")
            driver.get(url)
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Scroll down to load dynamic content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Give time for JavaScript to execute
            
            page_source = driver.page_source
            return BeautifulSoup(page_source, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url} with Selenium: {e}")
            return None
        finally:
            driver.quit()
    
    return None

def parse_catalog_page(soup, solution_type):
    """Parse the catalog page to extract solutions and their properties."""
    solutions = []
    
    if soup is None:
        print("No soup provided to parse_catalog_page")
        return solutions
    
    # Debug page content
    print("Page Title:", soup.title.text if soup.title else "No title found")
    
    # First, let's try to find any tables on the page
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables on the page")
    
    # Look for pre-packaged job solutions or individual solutions in the page content
    if solution_type == 'packaged':
        search_text = 'Pre-packaged Job Solutions'
    else:
        search_text = 'Individual Test Solutions'
    
    # Try different approaches to find the table
    # 1. Look for table with the solution type in any heading nearby
    for table in tables:
        headers = []
        try:
            # Check previous siblings for the heading
            prev_siblings = list(table.previous_siblings)
            for sibling in prev_siblings:
                if sibling.name in ['h1', 'h2', 'h3'] and search_text in sibling.get_text():
                    print(f"Found {solution_type} table with heading")
                    headers = [th.get_text().strip() for th in table.find_all('th')]
                    break
            
            # If we found the right table
            if headers and any(search_text in header for header in headers):
                print(f"Headers found: {headers}")
                return parse_table(table, headers, solution_type)
        except Exception as e:
            print(f"Error checking table heading: {e}")
    
    # 2. Look for table with column headers matching expected values
    for table in tables:
        try:
            headers = [th.get_text().strip() for th in table.find_all('th')]
            print(f"Table headers: {headers}")
            
            # Check if this table has the expected headers
            if search_text in headers and 'Remote Testing' in headers and 'Test Type' in headers:
                print(f"Found {solution_type} table by headers")
                return parse_table(table, headers, solution_type)
        except Exception as e:
            print(f"Error checking table headers: {e}")
    
    # 3. If we still haven't found it, try parse the first table in the main content area
    main_content = soup.find('main') or soup.find('div', class_='main-content') or soup
    tables_in_main = main_content.find_all('table')
    if tables_in_main:
        print(f"Trying first table in main content area ({len(tables_in_main)} tables found)")
        try:
            headers = [th.get_text().strip() for th in tables_in_main[0].find_all('th')]
            return parse_table(tables_in_main[0], headers, solution_type)
        except Exception as e:
            print(f"Error parsing first table in main: {e}")
    
    # Debug: Save HTML for inspection
    debug_file = f"data/debug_{solution_type}_{int(time.time())}.html"
    os.makedirs(os.path.dirname(debug_file), exist_ok=True)
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(str(soup))
    print(f"Saved debug HTML to {debug_file}")
    
    print(f"Could not find {solution_type} solutions table")
    return solutions

def parse_table(table, headers, solution_type):
    """Parse a table given its headers."""
    solutions = []
    
    try:
        # Find indices for columns
        name_index = next((i for i, h in enumerate(headers) if 'solution' in h.lower() or 'test' in h.lower()), 0)
        remote_testing_index = next((i for i, h in enumerate(headers) if 'remote' in h.lower()), None)
        adaptive_index = next((i for i, h in enumerate(headers) if 'adaptive' in h.lower() or 'irt' in h.lower()), None)
        test_type_index = next((i for i, h in enumerate(headers) if 'test type' in h.lower()), None)
        
        print(f"Column indices: name={name_index}, remote={remote_testing_index}, adaptive={adaptive_index}, test_type={test_type_index}")
        
        # Process each row in the table
        rows = table.find_all('tr')
        print(f"Found {len(rows)} rows in table")
        
        for i, row in enumerate(rows[1:], 1):  # Skip header row
            try:
                cells = row.find_all('td')
                if len(cells) < len(headers):
                    print(f"Row {i} has fewer cells ({len(cells)}) than headers ({len(headers)}), skipping")
                    continue
                
                # Get solution name and URL
                name_cell = cells[name_index]
                name_link = name_cell.find('a')
                if not name_link:
                    print(f"No link found in row {i}, skipping")
                    continue
                
                solution = {
                    "name": name_link.get_text().strip(),
                    "url": urljoin(BASE_URL, name_link['href']),
                    "solution_type": "packaged" if solution_type == 'packaged' else "individual",
                    "remote_testing": False,
                    "adaptive_irt": False,
                    "test_type": [],
                    "test_type_details": {}
                }
                
                # Add remote testing, adaptive/IRT and test type if available
                if remote_testing_index is not None:
                    solution["remote_testing"] = bool(cells[remote_testing_index].find('span', class_='check-mark') or '●' in cells[remote_testing_index].get_text())
                
                if adaptive_index is not None:
                    solution["adaptive_irt"] = bool(cells[adaptive_index].find('span', class_='check-mark') or '●' in cells[adaptive_index].get_text())
                
                if test_type_index is not None:
                    test_types = [t.strip() for t in cells[test_type_index].get_text().strip().split() if t.strip() in ['A', 'B', 'C', 'D', 'P', 'K', 'S']]
                    for test_type in test_types:
                        solution["test_type"].append(test_type)
                        solution["test_type_details"][test_type] = {
                            'name': '',  # Will be filled in from detail page
                            'description': ''  # Will be filled in from detail page
                        }
                
                print(f"Parsed solution: {solution['name']}")
                solutions.append(solution)
            except Exception as e:
                print(f"Error parsing row {i}: {e}")
    
    except Exception as e:
        print(f"Error parsing table: {e}")
    
    print(f"Found {len(solutions)} solutions in table")
    return solutions

def parse_solution_details(url):
    """Parse the details page of a solution."""
    # Try with requests first, if it fails use Selenium
    soup = get_page_content(url)
    if not soup or not soup.find(['h1', 'h2', 'h3']):
        # If no headings found, the page might be dynamically rendered - try with Selenium
        soup = get_page_content(url, use_selenium=True)
    
    if not soup:
        return {}
    
    # Define test type mapping
    test_type_mapping = {
        'A': 'Ability & Aptitude',
        'B': 'Biodata & Situational Judgement',
        'C': 'Competencies',
        'D': 'Development & 360',
        'P': 'Assessment Exercises',
        'K': 'Knowledge & Skills',
        'S': 'Personality & Behavior'
    }
    
    solution_detail = {
        "description": "",
        "job_levels": [],
        "languages": [],
        "assessment_length": "",
        "remote_testing": False,
        "adaptive_irt": False,
        "test_type": [],
        "test_type_details": {},  # Store detailed test type information
        "test_type_names": []     # Store mapped test type names
    }
    
    try:
        # Debug info
        print(f"Solution page title: {soup.title.text if soup.title else 'No title'}")
        
        # Extract the description
        description_found = False
        current_test_type = None
        
        # Try finding by heading text
        description_section = soup.find(['h1', 'h2', 'h3'], string='Description')
        if description_section:
            description_found = True
            description_paragraphs = description_section.find_next_siblings('p')
            solution_detail['description'] = ' '.join([p.get_text().strip() for p in description_paragraphs if p.get_text().strip()])
        
        # Try finding by class
        if not description_found:
            description_section = soup.find(class_=lambda c: c and 'description' in c.lower())
            if description_section:
                description_found = True
                solution_detail['description'] = description_section.get_text().strip()
        
        # Try finding content after a Description keyword anywhere
        if not description_found:
            for element in soup.find_all(['p', 'div']):
                if 'Description' in element.get_text():
                    description_found = True
                    next_elements = []
                    current = element.next_sibling
                    while current and current.name != 'h1' and current.name != 'h2' and current.name != 'h3':
                        if current.name == 'p':
                            next_elements.append(current.get_text().strip())
                        current = current.next_sibling
                    
                    solution_detail['description'] = ' '.join(next_elements)
                    break
        
        # Extract job levels
        job_levels_section = soup.find(['h1', 'h2', 'h3'], string=lambda s: s and 'job level' in s.lower())
        if job_levels_section:
            job_levels = job_levels_section.find_next_sibling('p')
            if job_levels:
                solution_detail['job_levels'] = [level.strip() for level in job_levels.get_text().split(',')]
        
        # Extract languages
        languages_section = soup.find(['h1', 'h2', 'h3'], string=lambda s: s and 'language' in s.lower())
        if languages_section:
            languages = languages_section.find_next_sibling('p')
            if languages:
                solution_detail['languages'] = [lang.strip() for lang in languages.get_text().split(',')]
        
        # Extract assessment length
        assessment_section = soup.find(['h1', 'h2', 'h3'], string=lambda s: s and 'assessment length' in s.lower())
        if assessment_section:
            assessment_length = assessment_section.find_next_sibling('p')
            if assessment_length:
                solution_detail['assessment_length'] = assessment_length.get_text().strip()
                # Try to extract minutes
                solution_detail['duration_minutes'] = extract_duration_minutes(assessment_length.get_text())
        
        # Another approach for assessment length - look for text containing "minutes"
        if not solution_detail['assessment_length']:
            for p in soup.find_all('p'):
                if 'minute' in p.text.lower() and ('completion time' in p.text.lower() or 'assessment length' in p.text.lower()):
                    solution_detail['assessment_length'] = p.text.strip()
                    solution_detail['duration_minutes'] = extract_duration_minutes(p.text)
                    break
        
        # Extract test types and their details
        test_type_section = soup.find(['h1', 'h2', 'h3'], string=lambda s: s and 'test type' in s.lower())
        if test_type_section:
            # Look for test type indicators (A, B, C, D, P, K, S)
            for element in test_type_section.find_all_next(['p', 'div']):
                text = element.get_text().strip()
                if any(test_type in text for test_type in ['A', 'B', 'C', 'D', 'P', 'K', 'S']):
                    # Found a test type line
                    test_type = text.split()[0]  # Get the test type letter
                    if test_type in ['A', 'B', 'C', 'D', 'P', 'K', 'S']:
                        solution_detail['test_type'].append(test_type)
                        # Add the mapped name to test_type_names
                        solution_detail['test_type_names'].append(test_type_mapping.get(test_type, test_type))
                        current_test_type = test_type
                        solution_detail['test_type_details'][test_type] = {
                            'name': test_type_mapping.get(test_type, 
                                   text.split('-')[1].strip() if '-' in text else text.strip()),
                            'description': ''
                        }
                elif current_test_type and text and not text.startswith(('A', 'B', 'C', 'D', 'P', 'K', 'S')):
                    # Add description for current test type
                    solution_detail['test_type_details'][current_test_type]['description'] += text + " "
        
        # If no test types were found on the detail page, try to extract from any element that mentions test types
        if not solution_detail['test_type']:
            for element in soup.find_all(['p', 'div', 'span']):
                text = element.get_text().strip()
                for test_type in ['A', 'B', 'C', 'D', 'P', 'K', 'S']:
                    if test_type in text and (test_type + ' -' in text or test_type + ':' in text):
                        if test_type not in solution_detail['test_type']:
                            solution_detail['test_type'].append(test_type)
                            solution_detail['test_type_names'].append(test_type_mapping.get(test_type, test_type))
                            # Get the description after the test type letter
                            description = text.split(test_type + ' -', 1)[1].strip() if test_type + ' -' in text else ""
                            if not description and test_type + ':' in text:
                                description = text.split(test_type + ':', 1)[1].strip()
                            
                            solution_detail['test_type_details'][test_type] = {
                                'name': test_type_mapping.get(test_type, test_type),
                                'description': description
                            }
        
        # Check for Remote Testing using multiple approaches
        # 1. Look for text containing "Remote Testing"
        solution_detail['remote_testing'] = bool(soup.find(string=lambda s: s and 'remote testing' in s.lower()))
        
        # 2. Look for elements with specific classes that indicate remote testing
        if not solution_detail['remote_testing']:
            # Check for elements with "yes" in class name near "Remote Testing" text
            remote_indicators = soup.find_all(class_=lambda c: c and ('yes' in c.lower() or 'circle' in c.lower() or 'check' in c.lower()))
            for indicator in remote_indicators:
                if indicator.parent and 'remote testing' in indicator.parent.text.lower():
                    solution_detail['remote_testing'] = True
                    break
        
        # 3. Look for catalogue__circle with -yes class or similar indicators
        if not solution_detail['remote_testing']:
            remote_circles = soup.find_all(class_=lambda c: c and 'circle' in c.lower() and 'yes' in c.lower())
            for circle in remote_circles:
                parent_text = circle.parent.text.lower() if circle.parent else ""
                if 'remote testing' in parent_text:
                    solution_detail['remote_testing'] = True
                    break
        
        # Check for Adaptive/IRT using multiple approaches
        # 1. Look for text containing "adaptive" or "IRT"
        solution_detail['adaptive_irt'] = bool(soup.find(string=lambda s: s and ('adaptive' in s.lower() or 'irt' in s.lower())))
        
        # 2. Look for elements with specific classes that indicate adaptive/IRT
        if not solution_detail['adaptive_irt']:
            # Check for elements with "yes" in class name near "Adaptive" or "IRT" text
            adaptive_indicators = soup.find_all(class_=lambda c: c and ('yes' in c.lower() or 'circle' in c.lower() or 'check' in c.lower()))
            for indicator in adaptive_indicators:
                if indicator.parent and ('adaptive' in indicator.parent.text.lower() or 'irt' in indicator.parent.text.lower()):
                    solution_detail['adaptive_irt'] = True
                    break
        
        # 3. Look for catalogue__circle with -yes class or similar indicators
        if not solution_detail['adaptive_irt']:
            adaptive_circles = soup.find_all(class_=lambda c: c and 'circle' in c.lower() and 'yes' in c.lower())
            for circle in adaptive_circles:
                parent_text = circle.parent.text.lower() if circle.parent else ""
                if 'adaptive' in parent_text or 'irt' in parent_text:
                    solution_detail['adaptive_irt'] = True
                    break
        
        # Clean up the test type details
        for test_type in solution_detail['test_type_details']:
            solution_detail['test_type_details'][test_type]['description'] = solution_detail['test_type_details'][test_type]['description'].strip()
            # Ensure the test type name is set to the mapping
            solution_detail['test_type_details'][test_type]['name'] = test_type_mapping.get(test_type, 
                solution_detail['test_type_details'][test_type]['name'])
        
        # Clean up lists
        solution_detail['job_levels'] = [jl for jl in solution_detail['job_levels'] if jl]
        solution_detail['languages'] = [l for l in solution_detail['languages'] if l]
        
    except Exception as e:
        print(f"Error parsing solution details: {e}")
    
    print(f"Extracted details: {', '.join(solution_detail.keys())}")
    return solution_detail

def load_existing_solutions(type_name):
    """Load existing solutions from JSON files."""
    filename = f"data/{type_name}_solutions.json"
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading existing solutions: {e}")
    return []

def merge_solutions(existing, new):
    """Merge solutions, avoiding duplicates based on name."""
    if not existing:
        return new
        
    # Create a map of existing solutions by name
    solution_map = {s['name']: s for s in existing}
    
    # Add or update with new solutions
    for solution in new:
        if solution['name'] in solution_map:
            # Update existing solution with new details
            for key, value in solution.items():
                if key not in solution_map[solution['name']] or not solution_map[solution['name']][key]:
                    solution_map[solution['name']][key] = value
        else:
            # Add new solution
            solution_map[solution['name']] = solution
    
    return list(solution_map.values())

def scrape_shl_catalog():
    """Scrape the SHL product catalog for assessment details."""
    # Load existing data, if any
    packaged_solutions = load_existing_solutions("packaged")
    individual_solutions = load_existing_solutions("individual")
    
    print(f"Loaded {len(packaged_solutions)} existing packaged solutions and {len(individual_solutions)} individual solutions")
    
    # Scrape packaged solutions
    print("Scraping packaged solutions...")
    for url in PACKAGED_SOLUTIONS_PAGES:
        try:
            print(f"\nScraping {url}")
            soup = get_page_content(url)
            page_solutions = parse_catalog_page(soup, 'packaged')
            if page_solutions:
                print(f"Found {len(page_solutions)} packaged solutions on page")
                # Add the new solutions and remove duplicates
                packaged_solutions = merge_solutions(packaged_solutions, page_solutions)
                # Save progress after each page
                save_to_json(packaged_solutions, 'data/packaged_solutions.json')
                save_to_mongodb(packaged_solutions, MONGO_COLLECTION_PACKAGED)
            else:
                print(f"No packaged solutions found on {url}")
            # Be nice to the server
            time.sleep(2)  # Increased delay to avoid rate limiting
        except Exception as e:
            print(f"Error processing {url}: {e}")
            # Continue with next URL
    
    # Scrape individual solutions
    print("\nScraping individual solutions...")
    for url in INDIVIDUAL_SOLUTIONS_PAGES:
        try:
            print(f"\nScraping {url}")
            soup = get_page_content(url)
            page_solutions = parse_catalog_page(soup, 'individual')
            if page_solutions:
                print(f"Found {len(page_solutions)} individual solutions on page")
                # Add the new solutions and remove duplicates
                individual_solutions = merge_solutions(individual_solutions, page_solutions)
                # Save progress after each page
                save_to_json(individual_solutions, 'data/individual_solutions.json')
                save_to_mongodb(individual_solutions, MONGO_COLLECTION_INDIVIDUAL)
            else:
                print(f"No individual solutions found on {url}")
            # Be nice to the server
            time.sleep(2)  # Increased delay to avoid rate limiting
        except Exception as e:
            print(f"Error processing {url}: {e}")
            # Continue with next URL
    
    # If we have solutions, add detail information for each
    if packaged_solutions or individual_solutions:
        # First, process solutions that don't have details yet
        print("\nScraping details for solutions without details...")
        
        # Process packaged solutions
        solutions_to_process = [s for s in packaged_solutions if 'description' not in s]
        for i, solution in enumerate(solutions_to_process):
            try:
                print(f"\nScraping details for {solution['name']} ({i+1}/{len(solutions_to_process)})")
                details = parse_solution_details(solution['url'])
                solution.update(details)
                # Save progress periodically
                if (i + 1) % 10 == 0:
                    save_to_json(packaged_solutions, 'data/packaged_solutions.json')
                    save_to_mongodb(packaged_solutions, MONGO_COLLECTION_PACKAGED)
                # Be nice to the server
                time.sleep(2)  # Increased delay to avoid rate limiting
            except Exception as e:
                print(f"Error getting details for {solution['name']}: {e}")
                # Continue with next solution
        
        # Save final results for packaged solutions
        save_to_json(packaged_solutions, 'data/packaged_solutions.json')
        save_to_mongodb(packaged_solutions, MONGO_COLLECTION_PACKAGED)
        
        # Process individual solutions
        solutions_to_process = [s for s in individual_solutions if 'description' not in s]
        for i, solution in enumerate(solutions_to_process):
            try:
                print(f"\nScraping details for {solution['name']} ({i+1}/{len(solutions_to_process)})")
                details = parse_solution_details(solution['url'])
                solution.update(details)
                # Save progress periodically
                if (i + 1) % 10 == 0:
                    save_to_json(individual_solutions, 'data/individual_solutions.json')
                    save_to_mongodb(individual_solutions, MONGO_COLLECTION_INDIVIDUAL)
                # Be nice to the server
                time.sleep(2)  # Increased delay to avoid rate limiting
            except Exception as e:
                print(f"Error getting details for {solution['name']}: {e}")
                # Continue with next solution
        
        # Save final results for individual solutions
        save_to_json(individual_solutions, 'data/individual_solutions.json')
        save_to_mongodb(individual_solutions, MONGO_COLLECTION_INDIVIDUAL)
        
        print(f"\nScraped {len(packaged_solutions)} packaged solutions and {len(individual_solutions)} individual solutions")
        
        # Also merge and save as the original format for compatibility
        all_solutions = packaged_solutions + individual_solutions
        processed_df = preprocess_assessments(all_solutions)
        save_to_json(processed_df, 'data/shl_assessments.json')
        save_to_mongodb(processed_df.to_dict('records'), MONGO_COLLECTION)
    else:
        print("No solutions found. Check if the website structure has changed.")
    
    return {
        'packaged': packaged_solutions,
        'individual': individual_solutions
    }

def preprocess_assessments(assessments):
    """Clean and preprocess assessment data."""
    df = pd.DataFrame(assessments)
    
    # Fill NA values properly based on data type
    for col in df.columns:
        if col == 'description':
            df[col] = df[col].fillna('')
        elif col in ['remote_testing', 'adaptive_irt']:
            df[col] = df[col].fillna(False)
        elif col in ['job_levels', 'languages', 'test_type']:
            # For list columns, replace NA with empty list
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    
    # Convert duration to minutes if not already present
    if 'duration_minutes' not in df.columns and 'assessment_length' in df.columns:
        df['duration_minutes'] = df['assessment_length'].apply(lambda x: extract_duration_minutes(x) if x else None)
    
    # Map test type letters to names
    test_type_mapping = {
        'A': 'Ability & Aptitude',
        'B': 'Biodata & Situational Judgement',
        'C': 'Competencies',
        'D': 'Development & 360',
        'P': 'Assessment Exercises',
        'K': 'Knowledge & Skills',
        'S': 'Personality & Behavior'
    }
    
    # Create a new column with mapped test type names
    df['test_type_names'] = df['test_type'].apply(
        lambda types: [test_type_mapping.get(t, t) for t in types] if isinstance(types, list) else []
    )
    
    # Create text for embedding
    df['text_for_embedding'] = df.apply(
        lambda row: f"{row.get('name', '')} {' '.join(row.get('test_type_names', []))} " +
                   f"Duration: {row.get('assessment_length', '')} " +
                   f"Remote Testing: {row.get('remote_testing', False)} " +
                   f"Adaptive: {row.get('adaptive_irt', False)} " +
                   f"Description: {row.get('description', '')[:200]}",
        axis=1
    )
    
    return df

def extract_duration_minutes(duration_text):
    """Extract numerical duration in minutes from text."""
    if not duration_text:
        return None
        
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

def save_to_json(data, filename='data/shl_assessments.json'):
    """Save processed data to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if isinstance(data, pd.DataFrame):
        json_data = data.to_json(orient='records', indent=4)
        with open(filename, 'w') as f:
            f.write(json_data)
    else:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
            
    print(f"Data saved to {filename}")

def save_to_mongodb(data, collection_name=MONGO_COLLECTION):
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

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    scrape_shl_catalog()