import data_collection
import json
import pprint

def test_assessment_details():
    """Test parsing of assessment details from an example URL."""
    # URL to test
    url = "https://www.shl.com/solutions/products/product-catalog/view/insurance-account-manager-solution/"
    
    print(f"Testing assessment details for: {url}")
    
    # Get the details
    details = data_collection.parse_solution_details(url)
    
    # Check for key properties
    important_properties = [
        "description", 
        "job_levels", 
        "languages", 
        "assessment_length", 
        "duration_minutes",
        "remote_testing", 
        "adaptive_irt", 
        "test_type", 
        "test_type_names", 
        "test_type_details"
    ]
    
    # Print property status
    print("\nChecking important properties:")
    for prop in important_properties:
        status = "✓ Found" if prop in details and details[prop] else "✗ Missing or empty"
        print(f"{prop}: {status}")
    
    # Pretty print the details
    print("\nDetailed assessment properties:")
    pprint.pprint(details)
    
    # Test mapping of test types
    if "test_type" in details and details["test_type"]:
        print("\nTest type mapping:")
        test_type_mapping = {
            'A': 'Ability & Aptitude',
            'B': 'Biodata & Situational Judgement',
            'C': 'Competencies',
            'D': 'Development & 360',
            'P': 'Assessment Exercises',
            'K': 'Knowledge & Skills',
            'S': 'Personality & Behavior'
        }
        
        for test_type in details["test_type"]:
            if test_type in test_type_mapping:
                print(f"{test_type} -> {test_type_mapping[test_type]}")
    
    # Test if duration minutes extraction works
    if "assessment_length" in details and details["assessment_length"]:
        print(f"\nAssessment length: {details['assessment_length']}")
        minutes = data_collection.extract_duration_minutes(details["assessment_length"])
        print(f"Extracted minutes: {minutes}")
    
    return details

def test_multiple_assessments():
    """Test parsing multiple assessments to ensure consistency."""
    urls = [
        "https://www.shl.com/solutions/products/product-catalog/view/insurance-account-manager-solution/",
        "https://www.shl.com/solutions/products/product-catalog/view/insurance-administrative-assistant-solution/"
    ]
    
    print("\n" + "="*50)
    print("Testing multiple assessments")
    print("="*50)
    
    results = []
    
    for url in urls:
        print(f"\nTesting: {url}")
        details = data_collection.parse_solution_details(url)
        print(f"Remote Testing: {details.get('remote_testing', False)}")
        print(f"Adaptive/IRT: {details.get('adaptive_irt', False)}")
        print(f"Test Types: {details.get('test_type', [])}")
        print(f"Test Type Names: {details.get('test_type_names', [])}")
        print(f"Assessment Length: {details.get('assessment_length', '')}")
        print(f"Duration (minutes): {details.get('duration_minutes', None)}")
        
        # Count number of populated properties
        populated = sum(1 for prop in details if details[prop])
        print(f"Total populated properties: {populated}")
        
        results.append(details)
    
    # Save results to file for inspection
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTest results saved to test_results.json")
    
    return results

if __name__ == "__main__":
    print("="*50)
    print("TESTING DATA COLLECTION SCRIPT")
    print("="*50)
    
    # Test detailed assessment parsing
    single_result = test_assessment_details()
    
    # Test multiple assessments
    multiple_results = test_multiple_assessments()
    
    print("\nTest completed!") 