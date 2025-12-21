#!/usr/bin/env python3
"""
Simple filter testing script.
Run with: python test_filter.py
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

from utils.openai_queries import extract_filters_from_query
from utils.rag_filters import build_qdrant_filter
from endpoints.rag import search_cars
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
        """Define focused test cases - one per scenario type"""
        
        # ===== 1. EXACT FILTERS (should pass) =====
        self.test_cases.append(TestCase(
            name="Exact filters: Toyota Camry XV50 under 15M",
            query="I want to buy toyota camry xv 50 and my budget is under 15 000 000 tenge",
            expected_filters={"model": "Toyota Camry", "max_price": 15000000},
            should_find_results=True
        ))
        
        # ===== 2. FILTERING IN ANY LANGUAGE (English colors) =====
        self.test_cases.append(TestCase(
            name="Any language: Lexus RX gray (English)",
            query="Can you offer me any gray color lexus rx?",
            expected_filters={"model": "Lexus RX", "color": "gray"},  # серый металлик exists
            should_find_results=True
        ))
        
        # ===== 3. FILTERING WITH SYNONYMS (серого цвета = серый металлик) =====
        self.test_cases.append(TestCase(
            name="Color synonym: Lexus RX серого цвета",
            query="I am searching for lexus rx серого цвета",
            expected_filters={"model": "Lexus RX", "color": "серый"},  # Should match серый металлик
            should_find_results=True
        ))
        
        # ===== 4. GENERAL NATURAL LANGUAGE QUERY (semantic search should work) =====
        self.test_cases.append(TestCase(
            name="Natural language: Comfortable family offroad car",
            query="I want to buy a car which will be comfortable for family use and will allow to use it on offroad",
            expected_filters={},  # No specific filters, semantic search should find results
            should_find_results=True  # Semantic search should return relevant cars
        ))
        
        # ===== 5. INCORRECT CITY NAME (should correct) =====
        self.test_cases.append(TestCase(
            name="City typo: Toyota Camry in Almata",
            query="Offer me toyota camry in Almata",
            expected_filters={"model": "Toyota Camry", "city": "Алматы"},  # Should correct Almata -> Алматы
            should_find_results=True
        ))
        
        # ===== ADDITIONAL SCENARIOS =====
        
        # Color in Russian (exact match)
        self.test_cases.append(TestCase(
            name="Color RU: Subaru Outback черный",
            query="Найди черный Subaru Outback",
            expected_filters={"model": "Subaru Outback", "color": "черный"},
            should_find_results=True
        ))
        
        # City with more typos
        self.test_cases.append(TestCase(
            name="City typo: Toyota Camry in Алма-ата",
            query="Find toyota camry in Алма-ата",
            expected_filters={"model": "Toyota Camry", "city": "Алматы"},
            should_find_results=True
        ))
        
        # Model + Color + Price combination
        self.test_cases.append(TestCase(
            name="Complex: Honda Accord red under 2M",
            query="Find a red Honda Accord under 2 million tenge",
            expected_filters={"model": "Honda Accord", "color": "red", "max_price": 2000000},
            should_find_results=True
        ))
        
        # Failed case: combination doesn't exist
        self.test_cases.append(TestCase(
            name="No results: Subaru Outback white (doesn't exist)",
            query="Find a white Subaru Outback",
            expected_filters={"model": "Subaru Outback", "color": "white"},
            should_find_results=False
        ))
    
    def calculate_filter_match(self, expected: Dict[str, Any], extracted: Dict[str, Any]) -> float:
        """Calculate how well extracted filters match expected (0.0 to 1.0)"""
        # Remove None values from extracted filters for comparison
        extracted_clean = {k: v for k, v in extracted.items() if v is not None}
        expected_clean = {k: v for k, v in expected.items() if v is not None}
        
        # If both are empty, perfect match
        if not expected_clean and not extracted_clean:
            return 1.0
        
        # If expected is empty but extracted has values, check if that's acceptable
        if not expected_clean:
            # For natural language queries, having no filters extracted is correct
            return 1.0 if not extracted_clean else 0.0
        
        # If extracted is empty but expected has values, no match
        if not extracted_clean:
            return 0.0
        
        # Color translation map (English to Russian)
        color_translations = {
            "white": "белый",
            "black": "черный",
            "red": "красный",
            "blue": "синий",
            "green": "зеленый",
            "yellow": "желтый",
            "grey": "серый",
            "gray": "серый",
            "silver": "серебристый",
            "brown": "коричневый",
            "bronze": "бронза",
            "beige": "бежевый",
            "gold": "золотистый",
            "turquoise": "бирюзовый"
        }
        
        matches = 0
        total = 0
        
        for key, expected_value in expected_clean.items():
            total += 1
            extracted_value = extracted_clean.get(key)
            
            if expected_value is None:
                if extracted_value is None:
                    matches += 1
            elif extracted_value is not None:
                # String comparison (case-insensitive, strip whitespace)
                if isinstance(expected_value, str) and isinstance(extracted_value, str):
                    expected_lower = expected_value.lower().strip()
                    extracted_lower = extracted_value.lower().strip()
                    
                    # Exact match
                    if expected_lower == extracted_lower:
                        matches += 1
                    # For color field, check translations (English ↔ Russian)
                    elif key == "color":
                        # Check if extracted English color translates to expected Russian
                        if extracted_lower in color_translations:
                            if color_translations[extracted_lower] == expected_lower:
                                matches += 1
                            # Check for partial match (e.g., "черный металлик" contains "черный")
                            elif expected_lower in color_translations[extracted_lower] or \
                                 color_translations[extracted_lower] in expected_lower:
                                matches += 0.8
                        # Check if expected English color translates to extracted Russian
                        elif expected_lower in color_translations:
                            if color_translations[expected_lower] == extracted_lower:
                                matches += 1
                            elif extracted_lower in color_translations[expected_lower] or \
                                 color_translations[expected_lower] in extracted_lower:
                                matches += 0.8
                        # Partial match for fuzzy (e.g., "черный" matches "черный металлик")
                        elif expected_lower in extracted_lower or extracted_lower in expected_lower:
                            matches += 0.5
                    # Partial match for other string fields
                    elif expected_lower in extracted_lower or extracted_lower in expected_lower:
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
            chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
            extracted_filters = extract_filters_from_query(test.query, openai_client, chat_model)
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
            
            # Test logic:
            # 1. For natural language queries (expected_filters empty): Pass if results found matches should_find_results
            # 2. For filter queries: Pass if filter match > 50% AND results found > 0
            if not test.expected_filters or all(v is None for v in test.expected_filters.values()):
                # Natural language query: check if results match expectation
                test.passed = (test.results_count > 0) == test.should_find_results
            else:
                # Filter query: check filter match and results
                test.passed = test.filter_match_score > 0.5 and test.results_count > 0
            
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

