# Improvement of RAG-based AI System

## Overview

The main idea of this project is to create an AI assistant which can help clients to choose cars among all the database data according to client requests. Here are some important metrics for such a system:

- **Filter Completeness** - Percentage of relevant filters extracted from queries (no missing filters). Measures if system misses important filter information
- **Precision** - Percentage of returned results that are relevant to the query. Measures result quality - are we showing the right cars?
- **Fuzzy Matching** - Measures fuzzy queries matching effectiveness - a percentage of queries on natural language for which a result was returned according to user query
- **Response Completeness** - Percentage of responses that include all requested information (URLs, prices, etc.)
- **Average Response Time** - Measures system speed and user experience
- **Zero Results Rate** - Percentage of queries that return no results

It can be a lot of metrics if we dive deeper. From my point of view, some metrics can be summarized. The main result of system improvement is to offer a client some cars for their query, according to their request, even if they don't know at all which car they want to buy (filters), or made mistakes in their query, or asked to recommend a car according to their goals. It is a combination of metrics: filter completeness, fuzzy matching, zero result rate.

## Initial Test Cases

I have tried to create some test cases which will cover such cases, here is a list:

- Exact filters: Toyota Camry XV50 under 15M
- Any language: Lexus RX gray (English)
- Color synonym: Lexus RX серого цвета
- Natural language: Comfortable family offroad car
- City typo: Toyota Camry in Almata
- Color RU: Subaru Outback черный
- City typo: Toyota Camry in Алма-ата
- Complex: Honda Accord red under 2M
- No results: Subaru Outback white (doesn't exist)

## Initial System State

After running the test file (`test_filter.py`), I got the following results for the initial system state:

```
Total Tests: 9
Passed: 1 (11.1%)
Failed: 8 (88.9%)

Average Filter Match Score: 69.4%

Failed Tests:
  - Any language: Lexus RX gray (English): Filter=100.0%, Results=0
  - Color synonym: Lexus RX серого цвета: Filter=50.0%, Results=0
  - Natural language: Comfortable family offroad car: Filter=0.0%, Results=0
  - City typo: Toyota Camry in Almata: Filter=50.0%, Results=0
  - Color RU: Subaru Outback черный: Filter=100.0%, Results=0
  - City typo: Toyota Camry in Алма-ата: Filter=50.0%, Results=0
  - Complex: Honda Accord red under 2M: Filter=100.0%, Results=0
  - No results: Subaru Outback white (doesn't exist): Filter=100.0%, Results=0
```

Just 1 test passed - it was a request which was written in English language with exact filters and car model. This means that initially the system did not understand color synonyms, Russian language, and natural queries at all - which are the most valuable features for AI search - when you can write any query you want and get a result. That's why I decided to improve this.

## Improvements Made

### 1. Semantic Embeddings

I started with recreating embeddings. They were created in such way:

```
"{car['model']}, {car['generation']}, {car['mileage']}, {car['color']}, {car['engine']}, цена {car['price']}"
```

This way of creating embeddings is too strict. Embeddings must describe an entity, which will help to find necessary objects in the database by natural user query - for example - "Comfortable family offroad car". With such embeddings as I created first, it is impossible.

**Solution:** Create semantic descriptions for each car using AI and create embeddings using these descriptions - made changes in the ingest script and saved to the database.

Previously, the system asked a client to specify an exact car model to search. After creating semantic descriptions, I removed this logic.

### 2. Dynamic Similarity Threshold

Also, the similarity threshold was set as a static value. But I encountered a problem with it - when filters are not specified, too high value of similarity threshold did not allow to get any results. **Solution:** Improved it by calculating the threshold dynamically depending on the overall filters count.

### 3. Color and City Normalization

As a next step, I wanted to improve searching by color variations, filters on different languages, and cities with typos. It was necessary to work with the initial data again.

**Solution:** I added a `synonyms.py` file with necessary synonyms and recreated the database collection with these changes. Also, normalization applies while handling user search. For example, color 'красный металлик' will be converted to just 'red' and will filter the collection by the new color parameter, which is normalized too.

## Results

After implementing all improvements, the test results show significant improvement:

```
Total Tests: 9
Passed: 9 (100.0%)
Failed: 0 (0.0%)

Average Filter Match Score: 100.0%

IMPROVEMENT SUMMARY
Initial Pass Rate: 11.1% (1/9 tests)
Current Pass Rate: 100.0% (9/9 tests)
Improvement: +88.9 percentage points

✅ System improved by 88.9 percentage points
```

### Key Achievements

The system now successfully handles all test cases, including:
- ✅ Multi-language queries (English and Russian)
- ✅ Color synonyms and variations (e.g., "серого цвета" → "gray")
- ✅ City name typos and variations (e.g., "Almata" → "Алматы")
- ✅ Natural language queries (e.g., "Comfortable family offroad car")
- ✅ Complex filter combinations (multiple filters in one query)

The improvement from 11.1% to 100.0% pass rate, demonstrates that all the implemented enhancements (semantic embeddings, dynamic similarity threshold, and color/city normalization) have successfully addressed the initial system limitations.

