#!/usr/bin/env python3
"""
Simple filter testing script.
Run with: python test_filter.py
"""
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from endpoints.rag import extract_filters_from_query, build_qdrant_filter, search_cars
from models.models import RagQueryRequest
from loaders import init_openai_client, init_qdrant_client, get_openai_client, get_qdrant_client
from qdrant_client.models import Filter


class TestCase:
    """Simple test case"""
    def __init__(self, name: str, query: str, expected_filters: Dict[str, Any], should_find_results: bool = True):
        self.name = name
        self.query = query
        self.expected_filters = expected_filters
        self.should_find_results = should_find_results
        self.passed = False
        self.error = None
        self.extracted_filters = {}
        self.results_count = 0
        self.filter_match_score = 0.0


class FilterTester:
    """Simple filter tester"""
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.setup_tests()
    
    def setup_tests(self):
        """Define test cases"""
        
        # Exact color filter tests (should work now)
        self.test_cases.append(TestCase(
            name="Exact color: черный",
            query="Find me a black car",
            expected_filters={"color": "черный"},
            should_find_results=True
        ))
        
        self.test_cases.append(TestCase(
            name="Exact color: белый",
            query="I want a white car",
            expected_filters={"color": "белый"},
            should_find_results=True
        ))
        
        self.test_cases.append(TestCase(
            name="Exact color: красный",
            query="Show me red cars",
            expected_filters={"color": "красный"},
            should_find_results=True
        ))
        
        self.test_cases.append(TestCase(
            name="Exact color: синий",
            query="Find blue cars",
            expected_filters={"color": "синий"},
            should_find_results=True
        ))
        
        # Fuzzy color tests (should work after improvements)
        self.test_cases.append(TestCase(
            name="Fuzzy color: black (English)",
            query="Find a black car",
            expected_filters={"color": "черный"},  # Should match "черный" or "черный металлик"
            should_find_results=True
        ))
        
        self.test_cases.append(TestCase(
            name="Fuzzy color: silver (English)",
            query="I want a silver car",
            expected_filters={"color": "серебристый"},  # Should match "серебристый" or "серебристый металлик"
            should_find_results=True
        ))
        
        self.test_cases.append(TestCase(
            name="Fuzzy color: grey vs серый",
            query="Find grey cars",
            expected_filters={"color": "серый"},  # Should match "серый" or "серый металлик"
            should_find_results=True
        ))
        
        # Model + Color combinations
        self.test_cases.append(TestCase(
            name="Model + Exact color",
            query="Find a white Subaru Outback",
            expected_filters={"model": "Subaru Outback", "color": "белый"},
            should_find_results=True
        ))
        
        self.test_cases.append(TestCase(
            name="Model + Fuzzy color",
            query="Find a black Honda Accord",
            expected_filters={"model": "Honda Accord", "color": "черный"},
            should_find_results=True
        ))
        
        # Price filters
        self.test_cases.append(TestCase(
            name="Price filter",
            query="Cars under 5 million tenge",
            expected_filters={"max_price": 5000000},
            should_find_results=True
        ))
        
        # Complex filters
        self.test_cases.append(TestCase(
            name="Complex: Model + Color + Price",
            query="Find a white Subaru Outback under 2 million tenge",
            expected_filters={"model": "Subaru Outback", "color": "белый", "max_price": 2000000},
            should_find_results=True
        ))
        
        self.test_cases.append(TestCase(
            name="Complex: Fuzzy color + Price",
            query="Find a black car under 3 million",
            expected_filters={"color": "черный", "max_price": 3000000},
            should_find_results=True
        ))
        
        # City filters (exact)
        self.test_cases.append(TestCase(
            name="City filter",
            query="Cars in Алматы",
            expected_filters={"city": "Алматы"},
            should_find_results=True
        ))
        
        # Fuzzy city (should work after improvements)
        self.test_cases.append(TestCase(
            name="Fuzzy city: Almaty (English)",
            query="Cars in Almaty",
            expected_filters={"city": "Алматы"},
            should_find_results=True
        ))
        
        # Engine filters
        self.test_cases.append(TestCase(
            name="Engine: hybrid",
            query="Find hybrid cars",
            expected_filters={"engine": "гибрид"},
            should_find_results=True
        ))
        
        # Color variations (testing fuzzy matching)
        self.test_cases.append(TestCase(
            name="Color variation: черный металлик",
            query="Find a black metallic car",
            expected_filters={"color": "черный металлик"},
            should_find_results=True
        ))
        
        self.test_cases.append(TestCase(
            name="Color variation: белый металлик",
            query="I want a white metallic car",
            expected_filters={"color": "белый металлик"},
            should_find_results=True
        ))
        
        # Edge case: typo in color
        self.test_cases.append(TestCase(
            name="Typo tolerance: blak (typo)",
            query="Find a blak car",
            expected_filters={"color": "черный"},  # Should handle typo
            should_find_results=True
        ))
    
    def calculate_filter_match(self, expected: Dict[str, Any], extracted: Dict[str, Any]) -> float:
        """Calculate how well extracted filters match expected (0.0 to 1.0)"""
        if not expected and not extracted:
            return 1.0
        
        if not expected or not extracted:
            return 0.0
        
        matches = 0
        total = 0
        
        for key, expected_value in expected.items():
            total += 1
            extracted_value = extracted.get(key)
            
            if expected_value is None:
                if extracted_value is None:
                    matches += 1
            elif extracted_value is not None:
                # String comparison (case-insensitive, strip whitespace)
                if isinstance(expected_value, str) and isinstance(extracted_value, str):
                    if expected_value.lower().strip() == extracted_value.lower().strip():
                        matches += 1
                    # Partial match for fuzzy (e.g., "черный" matches "черный металлик")
                    elif expected_value.lower().strip() in extracted_value.lower().strip() or \
                         extracted_value.lower().strip() in expected_value.lower().strip():
                        matches += 0.5
                # Numeric comparison (with 5% tolerance)
                elif isinstance(expected_value, (int, float)) and isinstance(extracted_value, (int, float)):
                    tolerance = abs(expected_value) * 0.05
                    if abs(expected_value - extracted_value) <= tolerance:
                        matches += 1
                # Exact match for other types
                elif expected_value == extracted_value:
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
    async def run_test(self, test: TestCase) -> bool:
        """Run a single test case"""
        try:
            openai_client = get_openai_client()
            
            # Extract filters
            extracted_filters = extract_filters_from_query(test.query, openai_client)
            test.extracted_filters = extracted_filters
            
            # Calculate filter match score
            test.filter_match_score = self.calculate_filter_match(test.expected_filters, extracted_filters)
            
            # Run search to see if we get results
            request = RagQueryRequest(question=test.query, top_k=5)
            response = await search_cars(request)
            
            # Count URLs in response (rough estimate of results)
            import re
            urls = re.findall(r'https?://[^\s]+', response.data)
            test.results_count = len(urls)
            
            # Test passes if:
            # 1. Filter match score is reasonable (> 0.5)
            # 2. If should_find_results, we got some results
            filter_ok = test.filter_match_score >= 0.5
            results_ok = not test.should_find_results or test.results_count > 0
            
            test.passed = filter_ok and results_ok
            
            return test.passed
            
        except Exception as e:
            test.error = str(e)
            test.passed = False
            return False
    
    async def run_all_tests(self):
        """Run all test cases"""
        print("=" * 70)
        print("FILTER TESTING")
        print("=" * 70)
        print(f"Running {len(self.test_cases)} test cases...\n")
        
        passed = 0
        failed = 0
        
        for i, test in enumerate(self.test_cases, 1):
            print(f"[{i}/{len(self.test_cases)}] {test.name}")
            print(f"  Query: {test.query}")
            
            success = await self.run_test(test)
            
            if success:
                passed += 1
                status = "✅ PASS"
            else:
                failed += 1
                status = "❌ FAIL"
            
            print(f"  Status: {status}")
            print(f"  Filter Match: {test.filter_match_score:.1%}")
            print(f"  Results Found: {test.results_count}")
            
            if test.error:
                print(f"  Error: {test.error}")
            
            # Show filter comparison
            if test.expected_filters:
                print(f"  Expected: {test.expected_filters}")
            if test.extracted_filters:
                print(f"  Extracted: {test.extracted_filters}")
            
            print()
        
        # Summary
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {len(self.test_cases)}")
        print(f"Passed: {passed} ({passed/len(self.test_cases)*100:.1f}%)")
        print(f"Failed: {failed} ({failed/len(self.test_cases)*100:.1f}%)")
        print()
        
        # Filter accuracy breakdown
        avg_filter_match = sum(t.filter_match_score for t in self.test_cases) / len(self.test_cases)
        print(f"Average Filter Match Score: {avg_filter_match:.1%}")
        print()
        
        # Show failed tests
        if failed > 0:
            print("Failed Tests:")
            for test in self.test_cases:
                if not test.passed:
                    print(f"  - {test.name}: Filter={test.filter_match_score:.1%}, Results={test.results_count}")
                    if test.error:
                        print(f"    Error: {test.error}")
            print()
        
        print("=" * 70)
        
        return passed, failed


async def main():
    """Main entry point"""
    # Initialize clients
    print("Initializing clients...")
    try:
        init_qdrant_client()
        init_openai_client()
        print("✅ Clients initialized\n")
    except Exception as e:
        print(f"❌ Failed to initialize clients: {e}")
        sys.exit(1)
    
    # Run tests
    tester = FilterTester()
    passed, failed = await tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())

