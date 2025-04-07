import os
import json
from data_collection import get_page_content, parse_solution_details, extract_duration_minutes

def test_assessment_duration_extraction():
    """Test the extraction of assessment duration from a specific URL."""
    # URL from the user's example
    url = "https://www.shl.com/solutions/products/product-catalog/view/account-manager-solution/"
    
    print(f"Testing duration extraction from: {url}")
    
    # Parse the solution details
    details = parse_solution_details(url)
    
    # Print the extracted information
    print("\nExtracted Details:")
    print(f"Description: {details.get('description', '')[:100]}...")
    print(f"Assessment Length: {details.get('assessment_length', '')}")
    print(f"Duration Minutes: {details.get('duration_minutes')}")
    print(f"Test Types: {details.get('test_type', [])}")
    print(f"Test Type Names: {details.get('test_type_names', [])}")
    print(f"Remote Testing: {details.get('remote_testing')}")
    print(f"Adaptive/IRT: {details.get('adaptive_irt')}")
    
    # Save the results to a JSON file for inspection
    os.makedirs("data", exist_ok=True)
    with open("data/test_extraction_results.json", "w") as f:
        json.dump(details, f, indent=4)
    
    print(f"\nResults saved to data/test_extraction_results.json")

if __name__ == "__main__":
    test_assessment_duration_extraction() 