import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
import time
from dotenv import load_dotenv
from pymongo import MongoClient
from urllib.parse import urljoin
import re
import traceback
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

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
                    "test_type_names": []
                }
                
                # Add remote testing, adaptive/IRT and test type if available
                if remote_testing_index is not None:
                    solution["remote_testing"] = bool(cells[remote_testing_index].find('span', class_='check-mark') or '●' in cells[remote_testing_index].get_text())
                
                if adaptive_index is not None:
                    solution["adaptive_irt"] = bool(cells[adaptive_index].find('span', class_='check-mark') or '●' in cells[adaptive_index].get_text())
                
                if test_type_index is not None:
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
                    
                    test_types = [t.strip() for t in cells[test_type_index].get_text().strip().split() if t.strip() in ['A', 'B', 'C', 'D', 'P', 'K', 'S']]
                    for test_type in test_types:
                        solution["test_type"].append(test_type)
                        solution["test_type_names"].append(test_type_mapping.get(test_type, test_type))
                
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
    # Always use Selenium for detail pages to ensure we get dynamic content
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
        "duration_minutes": None,
        "remote_testing": False,
        "adaptive_irt": False,
        "test_type": [],
        "test_type_names": []     # Store mapped test type names
    }
    
    try:
        # Debug info
        print(f"Solution page title: {soup.title.text if soup.title else 'No title'}")
        print(f"Parsing solution details from URL: {url}")
        
        # Extract the description - try multiple approaches
        description_found = False
        
        # 0. Look for the specific div class structure with XPath
        try:
            description_div = soup.find('div', class_='product-catalogue-training-calendar__row typ')
            if description_div:
                p = description_div.find('p')
                if p:
                    solution_detail['description'] = p.get_text().strip()
                    description_found = True
                    print("Found description in product-catalogue-training-calendar__row")
        except Exception as e:
            print(f"Error in XPath description extraction: {e}")
        
        # 1. Look for the main content div that might contain the description
        if not description_found:
            main_content = soup.find('div', class_=lambda c: c and ('main-content' in c.lower() or 'product-details' in c.lower()))
            if main_content:
                # Try to find a section with "Description" header or similar
                for header in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'strong']):
                    if 'description' in header.text.lower() or 'about this solution' in header.text.lower():
                        # Get the paragraphs after this header
                        description_text = []
                        next_elem = header.find_next(['p', 'div'])
                        while next_elem and next_elem.name in ['p', 'div'] and not next_elem.find(['h1', 'h2', 'h3', 'h4']):
                            description_text.append(next_elem.get_text().strip())
                            next_elem = next_elem.find_next_sibling()
                        
                        if description_text:
                            solution_detail['description'] = ' '.join(description_text)
                            description_found = True
                            print("Found description in main content section")
                            break
        
        # 2. Try finding by class or section containing "description"
        if not description_found:
            description_section = soup.find(class_=lambda c: c and 'description' in c.lower())
            if description_section:
                solution_detail['description'] = description_section.get_text().strip()
                description_found = True
                print("Found description in description section")
        
        # 3. Look for sections with specific pattern - Description: or Description followed by text
        if not description_found:
            for elem in soup.find_all(['p', 'div']):
                text = elem.get_text().strip()
                if text.startswith('Description:') or text.startswith('Description -') or 'solution description' in text.lower():
                    solution_detail['description'] = text.split(':', 1)[1].strip() if ':' in text else text.split('-', 1)[1].strip()
                    description_found = True
                    print("Found description in text pattern")
                    break
        
        # 4. Try to extract description from meta tags
        if not description_found:
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and 'content' in meta_desc.attrs:
                solution_detail['description'] = meta_desc['content']
                description_found = True
                print("Found description in meta tags")
        
        # 5. Look for the first substantial paragraph in the main content area
        if not description_found and main_content:
            paragraphs = main_content.find_all('p', recursive=False)
            if paragraphs and len(paragraphs[0].get_text().strip()) > 50:  # At least 50 chars to be substantial
                solution_detail['description'] = paragraphs[0].get_text().strip()
                description_found = True
                print("Found description in first substantial paragraph")
        
        # 6. As a last resort, try to find any paragraph with substantial content
        if not description_found:
            for p in soup.find_all('p'):
                text = p.get_text().strip()
                if len(text) > 100:  # Look for paragraphs with more than 100 characters
                    solution_detail['description'] = text
                    description_found = True
                    print("Found description in substantial paragraph")
                    break
        
        # Clean up the description
        if solution_detail['description']:
            # Remove extra whitespace
            solution_detail['description'] = ' '.join(solution_detail['description'].split())
            # Remove any HTML tags that might have been missed
            solution_detail['description'] = BeautifulSoup(solution_detail['description'], 'html.parser').get_text()
            print(f"Final description length: {len(solution_detail['description'])} characters")
        
        # Extract job levels - try multiple approaches
        # 1. Look for a section with "Job Levels" header
        job_levels_found = False
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong']):
            if 'job level' in header.text.lower() or 'position level' in header.text.lower():
                # Get the text after this header
                next_elem = header.find_next(['p', 'div', 'ul'])
                if next_elem:
                    # Handle different formats (comma-separated list or bullet points)
                    if next_elem.name == 'ul':
                        job_levels = [li.get_text().strip() for li in next_elem.find_all('li')]
                    else:
                        job_levels_text = next_elem.get_text().strip()
                        job_levels = [level.strip() for level in re.split(r'[,;•]', job_levels_text)]
                    
                    solution_detail['job_levels'] = [jl for jl in job_levels if jl]
                    job_levels_found = True
                    break
        
        # 2. Look for any element that mentions "Job Levels" followed by a list
        if not job_levels_found:
            for elem in soup.find_all(['p', 'div', 'td']):
                text = elem.get_text().strip()
                if text.startswith('Job levels:') or text.startswith('Job Levels:') or 'position levels:' in text.lower():
                    levels_text = text.split(':', 1)[1].strip()
                    solution_detail['job_levels'] = [level.strip() for level in re.split(r'[,;•]', levels_text)]
                    job_levels_found = True
                    break
        
        # 3. Look for common job level terms in any content
        if not job_levels_found:
            common_job_levels = ['Entry-Level', 'Mid-Professional', 'Senior-Professional', 'Manager', 'Executive']
            for elem in soup.find_all(['p', 'div', 'li']):
                text = elem.get_text().strip()
                for level in common_job_levels:
                    if level.lower() in text.lower() and level not in solution_detail['job_levels']:
                        solution_detail['job_levels'].append(level)
                        job_levels_found = True
        
        # Extract languages - try multiple approaches
        # 1. Look for a section with "Languages" header
        languages_found = False
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong']):
            if 'language' in header.text.lower() and 'programming' not in header.text.lower():
                # Get the text after this header
                next_elem = header.find_next(['p', 'div', 'ul'])
                if next_elem:
                    # Handle different formats (comma-separated list or bullet points)
                    if next_elem.name == 'ul':
                        languages = [li.get_text().strip() for li in next_elem.find_all('li')]
                    else:
                        languages_text = next_elem.get_text().strip()
                        languages = [lang.strip() for lang in re.split(r'[,;•]', languages_text)]
                    
                    solution_detail['languages'] = [lang for lang in languages if lang]
                    languages_found = True
                    break
        
        # 2. Look for any element that mentions "Languages" followed by a list
        if not languages_found:
            for elem in soup.find_all(['p', 'div', 'td']):
                text = elem.get_text().strip()
                if text.startswith('Languages:') or 'available in:' in text.lower() or 'available languages:' in text.lower():
                    langs_text = text.split(':', 1)[1].strip()
                    solution_detail['languages'] = [lang.strip() for lang in re.split(r'[,;•]', langs_text)]
                    languages_found = True
                    break
        
        # 3. Look for common language terms in any content
        if not languages_found:
            common_languages = ['English', 'Spanish', 'French', 'German', 'Chinese', 'Japanese']
            for elem in soup.find_all(['p', 'div', 'li']):
                text = elem.get_text().strip()
                for lang in common_languages:
                    if lang in text and '(' in text and ')' in text:  # Language often appears like "English (US)"
                        full_lang = text[text.find(lang):text.find(')', text.find(lang)) + 1].strip()
                        if full_lang and full_lang not in solution_detail['languages']:
                            solution_detail['languages'].append(full_lang)
                            languages_found = True
                    elif lang in text and lang not in solution_detail['languages']:
                        solution_detail['languages'].append(lang)
                        languages_found = True
        
        # Extract assessment length - improved method with more patterns
        assessment_found = False
        
        # 0. Look specifically for the pattern in the example: "Approximate Completion Time in minutes = X"
        for elem in soup.find_all('p'):
            text = elem.get_text().strip()
            if 'approximate completion time in minutes' in text.lower():
                print(f"Found completion time pattern: {text}")
                solution_detail['assessment_length'] = text
                minutes_match = re.search(r'=\s*(\d+)', text)
                if minutes_match:
                    solution_detail['duration_minutes'] = int(minutes_match.group(1))
                else:
                    solution_detail['duration_minutes'] = extract_duration_minutes(text)
                assessment_found = True
                break
        
        # 1. Look for a section with "Assessment Length" header
        if not assessment_found:
            for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong']):
                if 'assessment length' in header.text.lower() or 'completion time' in header.text.lower() or 'test duration' in header.text.lower():
                    # Get the text after this header
                    next_elem = header.find_next(['p', 'div'])
                    if next_elem:
                        text = next_elem.get_text().strip()
                        print(f"Found completion time via header: {text}")
                        solution_detail['assessment_length'] = text
                        solution_detail['duration_minutes'] = extract_duration_minutes(text)
                        assessment_found = True
                        break
        
        # 2. Look for any element that contains "Completion Time" or similar
        if not assessment_found:
            time_patterns = [
                'completion time', 'assessment length', 'test duration', 
                'approximate time', 'time to complete', 'takes approximately',
                'duration:', 'time:', 'minutes to complete'
            ]
            for elem in soup.find_all(['p', 'div', 'li', 'td', 'span']):
                text = elem.get_text().strip()
                if any(pattern in text.lower() for pattern in time_patterns):
                    print(f"Found completion time via keyword: {text}")
                    solution_detail['assessment_length'] = text
                    solution_detail['duration_minutes'] = extract_duration_minutes(text)
                    assessment_found = True
                    break
        
        # 3. Look specifically for time indications with minutes/hours
        if not assessment_found:
            time_regex = r'\b(\d+)\s*(minutes?|mins?|hours?|hrs?)\b'
            for elem in soup.find_all(['p', 'div', 'li', 'span']):
                text = elem.get_text().strip().lower()
                match = re.search(time_regex, text, re.IGNORECASE)
                if match and ('takes' in text or 'duration' in text or 'time' in text or 'complete' in text or 'approximately' in text):
                    print(f"Found completion time via regex: {text}")
                    solution_detail['assessment_length'] = text
                    solution_detail['duration_minutes'] = extract_duration_minutes(text)
                    assessment_found = True
                    break
        
        # Extract test types - modified approach to not create test_type_details
        # 1. Look for a section with "Test Type" header
        test_types_found = False
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong']):
            if 'test type' in header.text.lower() or 'assessment type' in header.text.lower():
                # Get all elements after this header that might contain test types
                current_elem = header.find_next_sibling()
                while current_elem and current_elem.name not in ['h1', 'h2', 'h3', 'h4']:
                    text = current_elem.get_text().strip()
                    
                    # Check for test type codes (A, B, C, D, P, K, S)
                    for test_type in ['A', 'B', 'C', 'D', 'P', 'K', 'S']:
                        if (test_type + ' -' in text or 
                            test_type + ':' in text or 
                            f"{test_type} " in text or 
                            f", {test_type}" in text or 
                            f"; {test_type}" in text):
                            if test_type not in solution_detail['test_type']:
                                solution_detail['test_type'].append(test_type)
                                solution_detail['test_type_names'].append(test_type_mapping.get(test_type, test_type))
                    
                    current_elem = current_elem.find_next_sibling()
                
                test_types_found = bool(solution_detail['test_type'])
                break
        
        # 2. Check for test types mentioned anywhere in the document
        if not test_types_found:
            for elem in soup.find_all(['p', 'div', 'li', 'td']):
                text = elem.get_text().strip()
                for test_type in ['A', 'B', 'C', 'D', 'P', 'K', 'S']:
                    if (test_type + ' -' in text or 
                        test_type + ':' in text or 
                        f"{test_type} " in text or 
                        f", {test_type}" in text or 
                        f"; {test_type}" in text or
                        test_type == text):
                        if test_type not in solution_detail['test_type']:
                            solution_detail['test_type'].append(test_type)
                            solution_detail['test_type_names'].append(test_type_mapping.get(test_type, test_type))
        
        # Check for Remote Testing indication 
        for elem in soup.find_all(['p', 'div', 'li', 'span']):
            text = elem.get_text().strip().lower()
            if 'remote testing' in text or 'remote proctoring' in text or 'remote assessment' in text:
                if 'supported' in text or 'available' in text or 'enabled' in text or 'yes' in text or '✓' in text:
                    solution_detail['remote_testing'] = True
                    break
        
        # Check for Adaptive/IRT indication
        for elem in soup.find_all(['p', 'div', 'li', 'span']):
            text = elem.get_text().strip().lower()
            if ('adaptive' in text or 'irt' in text or 'item response theory' in text) and 'testing' in text:
                if 'supported' in text or 'available' in text or 'enabled' in text or 'yes' in text or '✓' in text:
                    solution_detail['adaptive_irt'] = True
                    break
                    
        # Ensure we have default values for required fields
        # Ensure job_levels has at least one value
        if not solution_detail['job_levels']:
            solution_detail['job_levels'] = ["Mid-Professional"]  # Default
            
        # Ensure languages has at least one value
        if not solution_detail['languages']:
            solution_detail['languages'] = ["English (USA)"]  # Default
            
        # Log findings
        print(f"Extracted details for {url}:")
        print(f"  - Description: {len(solution_detail['description'])} chars")
        print(f"  - Assessment Length: {solution_detail['assessment_length']}")
        print(f"  - Duration Minutes: {solution_detail['duration_minutes']}")
        print(f"  - Test Types: {solution_detail['test_type']}")
        print(f"  - Remote Testing: {solution_detail['remote_testing']}")
        
    except Exception as e:
        print(f"Error parsing solution details: {e}")
        traceback.print_exc()
    
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
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for url in PACKAGED_SOLUTIONS_PAGES:
            futures.append(executor.submit(process_catalog_page, url, 'packaged'))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                page_solutions = future.result()
                if page_solutions:
                    print(f"Found {len(page_solutions)} packaged solutions on page")
                    # Label each solution with its type
                    for solution in page_solutions:
                        solution['category'] = 'Pre-packaged Job Solutions'
                    # Add the new solutions and remove duplicates
                    packaged_solutions = merge_solutions(packaged_solutions, page_solutions)
                    # Save progress after each page
                    save_to_json(packaged_solutions, 'data/packaged_solutions.json')
                    save_to_mongodb(packaged_solutions, MONGO_COLLECTION_PACKAGED)
            except Exception as e:
                print(f"Error processing page: {e}")
    
    # Scrape individual solutions
    print("\nScraping individual solutions...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for url in INDIVIDUAL_SOLUTIONS_PAGES:
            futures.append(executor.submit(process_catalog_page, url, 'individual'))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                page_solutions = future.result()
                if page_solutions:
                    print(f"Found {len(page_solutions)} individual solutions on page")
                    # Label each solution with its type
                    for solution in page_solutions:
                        solution['category'] = 'Individual Test Solutions'
                    # Add the new solutions and remove duplicates
                    individual_solutions = merge_solutions(individual_solutions, page_solutions)
                    # Save progress after each page
                    save_to_json(individual_solutions, 'data/individual_solutions.json')
                    save_to_mongodb(individual_solutions, MONGO_COLLECTION_INDIVIDUAL)
            except Exception as e:
                print(f"Error processing page: {e}")
    
    # Process details concurrently
    print("\nScraping detailed information for all solutions...")
    
    def process_solution_details(solution, solution_type):
        try:
            print(f"\nScraping details for {solution['name']}")
            details = parse_solution_details(solution['url'])
            
            # Update existing details with new information
            for key, value in details.items():
                # For non-empty values, replace existing data
                if value or key not in solution:
                    solution[key] = value
            
            # Ensure we have category and solution_type
            solution['category'] = 'Pre-packaged Job Solutions' if solution_type == 'packaged' else 'Individual Test Solutions'
            solution['solution_type'] = solution_type
            
            return solution
        except Exception as e:
            print(f"Error getting details for {solution['name']}: {e}")
            return solution
    
    # Process packaged solutions
    print(f"\nProcessing {len(packaged_solutions)} packaged solutions...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_solution_details, solution, 'packaged') for solution in packaged_solutions]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                packaged_solutions[i] = future.result()
                # Save progress periodically
                if (i + 1) % 5 == 0:
                    save_to_json(packaged_solutions, 'data/packaged_solutions.json')
                    save_to_mongodb(packaged_solutions, MONGO_COLLECTION_PACKAGED)
            except Exception as e:
                print(f"Error processing solution: {e}")
    
    # Process individual solutions
    print(f"\nProcessing {len(individual_solutions)} individual solutions...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_solution_details, solution, 'individual') for solution in individual_solutions]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                individual_solutions[i] = future.result()
                # Save progress periodically
                if (i + 1) % 5 == 0:
                    save_to_json(individual_solutions, 'data/individual_solutions.json')
                    save_to_mongodb(individual_solutions, MONGO_COLLECTION_INDIVIDUAL)
            except Exception as e:
                print(f"Error processing solution: {e}")
    
    # Save final results
    save_to_json(packaged_solutions, 'data/packaged_solutions.json')
    save_to_mongodb(packaged_solutions, MONGO_COLLECTION_PACKAGED)
    save_to_json(individual_solutions, 'data/individual_solutions.json')
    save_to_mongodb(individual_solutions, MONGO_COLLECTION_INDIVIDUAL)
    
    print(f"\nScraped {len(packaged_solutions)} packaged solutions and {len(individual_solutions)} individual solutions")
    
    # Also merge and save as the original format for compatibility
    all_solutions = packaged_solutions + individual_solutions
    processed_df = preprocess_assessments(all_solutions)
    save_to_json(processed_df.to_dict('records'), 'data/shl_assessments.json')
    save_to_mongodb(processed_df.to_dict('records'), MONGO_COLLECTION)
    
    return {
        'packaged': packaged_solutions,
        'individual': individual_solutions
    }

def process_catalog_page(url, solution_type):
    """Process a single catalog page."""
    try:
        print(f"\nScraping {url}")
        soup = get_page_content(url)
        page_solutions = parse_catalog_page(soup, solution_type)
        return page_solutions
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return []

def preprocess_assessments(assessments):
    """Clean and preprocess assessment data."""
    # First, ensure all records have consistent fields
    for solution in assessments:
        # Ensure all required fields exist
        solution.setdefault('description', '')
        solution.setdefault('remote_testing', False)
        solution.setdefault('adaptive_irt', False)
        solution.setdefault('job_levels', [])
        solution.setdefault('languages', [])
        solution.setdefault('test_type', [])
        solution.setdefault('test_type_names', [])
        solution.setdefault('assessment_length', '')
        
        # Remove empty test_type_details as requested
        if 'test_type_details' in solution:
            del solution['test_type_details']
    
    # Create DataFrame
    df = pd.DataFrame(assessments)
    
    # Map test type letters to names if not already mapped
    test_type_mapping = {
        'A': 'Ability & Aptitude',
        'B': 'Biodata & Situational Judgement',
        'C': 'Competencies',
        'D': 'Development & 360',
        'P': 'Assessment Exercises',
        'K': 'Knowledge & Skills',
        'S': 'Personality & Behavior'
    }
    
    # Create or update test_type_names with properly mapped values
    df['test_type_names'] = df['test_type'].apply(
        lambda types: [test_type_mapping.get(t, t) for t in types] if isinstance(types, list) else []
    )
    
    # Ensure proper types for all columns
    df['description'] = df['description'].fillna('').astype(str)
    df['remote_testing'] = df['remote_testing'].fillna(False).astype(bool)
    df['adaptive_irt'] = df['adaptive_irt'].fillna(False).astype(bool)
    
    # Handle list columns
    for col in ['job_levels', 'languages', 'test_type', 'test_type_names']:
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    
    # Extract duration in minutes from assessment_length
    df['duration_minutes'] = df['assessment_length'].apply(extract_duration_minutes)
    
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
    if not duration_text or not isinstance(duration_text, str):
        return None
        
    try:
        duration_text = duration_text.lower()
        
        # Direct pattern matching for "Approximate Completion Time in minutes = X"
        if 'completion time in minutes =' in duration_text:
            parts = duration_text.split('=')
            if len(parts) >= 2:
                minutes_str = parts[1].strip()
                minutes = ''.join(c for c in minutes_str if c.isdigit())
                if minutes:
                    return int(minutes)
        
        # Try with regex for same pattern
        completion_time_match = re.search(r'completion time in minutes\s*=\s*(\d+)', duration_text, re.IGNORECASE)
        if completion_time_match:
            return int(completion_time_match.group(1))
            
        # Also match "Approximate Completion Time: X minutes"
        completion_time_match2 = re.search(r'completion time:?\s*(\d+)\s*minutes?', duration_text, re.IGNORECASE)
        if completion_time_match2:
            return int(completion_time_match2.group(1))
        
        # Extract numbers followed by "minute" or "min"
        minute_pattern = r'(\d+)\s*(?:minute|min)'
        minute_matches = re.findall(minute_pattern, duration_text)
        if minute_matches:
            return int(minute_matches[0])
        
        # Handle ranges like "15-20 minutes"
        range_pattern = r'(\d+)\s*-\s*(\d+)\s*(?:minute|min)'
        range_match = re.search(range_pattern, duration_text)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            return (start + end) // 2  # Average of range
        
        # Handle hours and convert to minutes
        if 'hour' in duration_text:
            # Look for "X hour(s) Y minute(s)" pattern
            hours_pattern = r'(\d+)\s*hour'
            minutes_pattern = r'(\d+)\s*minute'
            
            hours_match = re.search(hours_pattern, duration_text)
            minutes_match = re.search(minutes_pattern, duration_text)
            
            total_minutes = 0
            if hours_match:
                total_minutes += int(hours_match.group(1)) * 60
            if minutes_match:
                total_minutes += int(minutes_match.group(1))
                
            if total_minutes > 0:
                return total_minutes
            
            # Just hours with no minutes specified
            hours_only = re.findall(r'(\d+)\s*hour', duration_text)
            if hours_only:
                return int(hours_only[0]) * 60
        
        # Look for any digits followed by min/minutes or preceded by timing words
        timing_words = ['approximately', 'about', 'around', 'takes', 'duration', 'time to complete']
        for word in timing_words:
            if word in duration_text:
                # Find a number after this word
                pattern = rf'{word}\s+(\d+)'
                match = re.search(pattern, duration_text)
                if match:
                    return int(match.group(1))
                
                # Try reverse - number before "minutes" after the timing word
                idx = duration_text.find(word)
                if idx >= 0:
                    rest = duration_text[idx:]
                    num_match = re.search(r'(\d+)\s*min', rest)
                    if num_match:
                        return int(num_match.group(1))
        
        # Last resort - extract any digits from the text that might be related to time
        digits = re.findall(r'\d+', duration_text)
        if digits and ('time' in duration_text or 'duration' in duration_text or 'minutes' in duration_text):
            return int(digits[0])
            
    except Exception as e:
        print(f"Error extracting duration from '{duration_text}': {e}")
    
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