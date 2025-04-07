import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from firecrawl import FirecrawlApp
import asyncio
import time
import nest_asyncio

# Apply nest_asyncio for Jupyter/interactive environments
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Test type mapping
TEST_TYPE_MAPPING = {
    'A': 'Ability & Aptitude',
    'B': 'Biodata & Situational Judgement',
    'C': 'Competencies',
    'D': 'Development & 360',
    'E': 'Assessment Exercises',
    'K': 'Knowledge & Skills',
    'P': 'Personality & Behavior',
    'S': 'Simulations'
}

class TestType(BaseModel):
    name: str
    description: Optional[str] = None

class SHLSolution(BaseModel):
    name: str
    url: str
    description: Optional[str] = None
    job_levels: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    assessment_length: Optional[str] = None
    remote_testing: bool = False
    adaptive_irt: bool = False
    test_types: List[TestType] = Field(default_factory=list)

async def extract_shl_solutions():
    """Extract SHL solutions using Firecrawl's extract endpoint."""
    if not os.getenv("FIRECRAWL_API_KEY"):
        print("Error: FIRECRAWL_API_KEY environment variable not set")
        return
    
    # Initialize Firecrawl
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    
    # Base URL for SHL catalog
    base_url = "https://www.shl.com/solutions/products/product-catalog/"
    
    # Create URLs for both packaged and individual solutions
    all_urls = [
        f"{base_url}?start={i*12}&type=2" for i in range(13)  # Packaged solutions
    ] + [
        f"{base_url}?start={i*12}&type=1" for i in range(32)  # Individual solutions
    ]
    
    # Define the extraction prompt
    prompt = """
    Extract the following information from each solution in the SHL catalog:
    1. Solution name
    2. Solution URL
    3. Description
    4. Job levels (comma-separated)
    5. Languages (comma-separated)
    6. Test types (A, B, C, D, E, K, P, S)
    7. Remote testing status (check for catalogue__circle -yes class)
    8. Adaptive/IRT status (check for catalogue__circle -yes class)
    9. Assessment length (click on test and extract from "Approximate Completion Time in minutes")
    """
    
    all_solutions = []
    
    # Process URLs in batches of 10
    for i in range(0, len(all_urls), 10):
        url_batch = all_urls[i:i+10]
        print(f"\nProcessing batch {i//10 + 1} of {(len(all_urls) + 9)//10}")
        
        try:
            # Start the extraction job for this batch
            extract_job = await app.batch_scrape_urls(
                url_batch,
                {
                    'formats': ['extract'],
                    'extract': {
                        'prompt': prompt,
                        'schema': SHLSolution.model_json_schema()
                    }
                }
            )
            
            if not extract_job or 'id' not in extract_job:
                print(f"Failed to start extraction job for batch {i//10 + 1}")
                continue
                
            job_id = extract_job['id']
            
            # Get the status of the extraction job
            while True:
                job_status = await app.check_batch_scrape_status(job_id)
                
                if job_status['status'] == 'completed':
                    # Process the results
                    results = job_status.get('data', [])
                    
                    for result in results:
                        if 'extract' in result:
                            solution_data = result['extract']
                            
                            # Map test type letters to full names
                            if 'test_types' in solution_data:
                                solution_data['test_types'] = [
                                    TestType(name=TEST_TYPE_MAPPING.get(t['name'], t['name']))
                                    for t in solution_data['test_types']
                                ]
                            
                            all_solutions.append(SHLSolution(**solution_data))
                    
                    print(f"Successfully processed {len(results)} solutions from batch {i//10 + 1}")
                    break
                elif job_status['status'] == 'failed':
                    print(f"Batch {i//10 + 1} extraction failed")
                    break
                else:
                    print(f"Progress: {job_status.get('progress', 0)}%")
                    await asyncio.sleep(2)  # Add small delay between status checks
            
            # Add delay between batches to avoid rate limiting
            await asyncio.sleep(5)
            
        except Exception as e:
            print(f"Error processing batch {i//10 + 1}: {e}")
            continue
    
    print(f"\nExtraction complete. Total solutions found: {len(all_solutions)}")
    return all_solutions

if __name__ == "__main__":
    solutions = asyncio.run(extract_shl_solutions())
    
    # Print the results
    for solution in solutions:
        print(f"\nSolution: {solution.name}")
        print(f"URL: {solution.url}")
        print(f"Description: {solution.description}")
        print(f"Job Levels: {', '.join(solution.job_levels)}")
        print(f"Languages: {', '.join(solution.languages)}")
        print(f"Assessment Length: {solution.assessment_length}")
        print(f"Remote Testing: {solution.remote_testing}")
        print(f"Adaptive/IRT: {solution.adaptive_irt}")
        print("Test Types:")
        for test_type in solution.test_types:
            print(f"  - {test_type.name}")
            if test_type.description:
                print(f"    Description: {test_type.description}") 