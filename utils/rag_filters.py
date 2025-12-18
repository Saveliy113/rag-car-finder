"""
RAG filter utilities for building Qdrant filters and sorting results.
"""
from typing import Dict, Any
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue


def build_qdrant_filter(filters: dict) -> Filter:
    """
    Build Qdrant Filter object from extracted filters
    Uses numeric fields: price_num, mileage_num, year_num
    """
    conditions = []
    
    # Price filter (using price_num field)
    if filters.get("max_price") is not None and filters.get("min_price") is not None:
        # Both min and max price
        conditions.append(
            FieldCondition(
                key="price_num",
                range=Range(gte=filters["min_price"], lte=filters["max_price"])
            )
        )
    elif filters.get("max_price") is not None:
        conditions.append(
            FieldCondition(
                key="price_num",
                range=Range(lte=filters["max_price"])
            )
        )
    elif filters.get("min_price") is not None:
        conditions.append(
            FieldCondition(
                key="price_num",
                range=Range(gte=filters["min_price"])
            )
        )
    
    # Mileage filter (using mileage_num field)
    if filters.get("max_mileage") is not None and filters.get("min_mileage") is not None:
        # Both min and max mileage
        conditions.append(
            FieldCondition(
                key="mileage_num",
                range=Range(gte=filters["min_mileage"], lte=filters["max_mileage"])
            )
        )
    elif filters.get("max_mileage") is not None:
        conditions.append(
            FieldCondition(
                key="mileage_num",
                range=Range(lte=filters["max_mileage"])
            )
        )
    elif filters.get("min_mileage") is not None:
        conditions.append(
            FieldCondition(
                key="mileage_num",
                range=Range(gte=filters["min_mileage"])
            )
        )
    
    # Year filter (using modelYear field)
    if filters.get("year_preference") and isinstance(filters["year_preference"], int):
        # Specific year
        conditions.append(
            FieldCondition(
                key="modelYear",
                match=MatchValue(value=filters["year_preference"])
            )
        )
        # "newest" and "oldest" will be handled by sorting after search
    
    # Color filter
    if filters.get("color"):
        conditions.append(
            FieldCondition(
                key="color",
                match=MatchValue(value=filters["color"])
            )
        )
    
    # City filter
    if filters.get("city"):
        conditions.append(
            FieldCondition(
                key="city",
                match=MatchValue(value=filters["city"])
            )
        )
    
    # Engine filter
    if filters.get("engine"):
        conditions.append(
            FieldCondition(
                key="engine",
                match=MatchValue(value=filters["engine"])
            )
        )
    
    if conditions:
        return Filter(must=conditions)
    return None


def count_filters(filters: Dict[str, Any]) -> int:
    """
    Count the number of non-null filters specified (excluding model).
    Model is not counted as it's required for vector search.
    """
    count = 0
    
    # Price filters (count as 1 if either min or max is specified)
    if filters.get("max_price") is not None or filters.get("min_price") is not None:
        count += 1
    
    # Mileage filters (count as 1 if either min or max is specified)
    if filters.get("max_mileage") is not None or filters.get("min_mileage") is not None:
        count += 1
    
    # Color filter
    if filters.get("color"):
        count += 1
    
    # City filter
    if filters.get("city"):
        count += 1
    
    # Engine filter
    if filters.get("engine"):
        count += 1
    
    # Year filter (only count if it's a specific year, not "newest"/"oldest")
    if filters.get("year_preference") and isinstance(filters["year_preference"], int):
        count += 1
    
    return count


def calculate_dynamic_similarity_threshold(
    filters: Dict[str, Any],
    base_threshold: float = 0.5,
    min_threshold: float = 0.3,
    filter_increment: float = 0.05
) -> float:
    """
    Calculate dynamic similarity threshold based on filter count.
    
    Strategy:
    - When no filters are specified (natural language query only): use min_threshold (more lenient)
    - As more filters are added: increase threshold (more strict)
    - Maximum threshold is base_threshold
    
    Args:
        filters: Dictionary of extracted filters
        base_threshold: Maximum threshold when many filters are specified (default: 0.5)
        min_threshold: Minimum threshold when no filters are specified (default: 0.3)
        filter_increment: How much to increase threshold per filter (default: 0.05)
    
    Returns:
        Calculated similarity threshold
    """
    filter_count = count_filters(filters)
    
    # Start with minimum threshold
    threshold = min_threshold
    
    # Increase threshold based on filter count
    # Each filter adds increment, but cap at base_threshold
    threshold = min(min_threshold + (filter_count * filter_increment), base_threshold)
    
    return threshold


def sort_results_by_year_preference(results, filters: dict):
    """Sort results by year preference (newest/oldest) if specified"""
    if not filters.get("year_preference"):
        return results
    
    if filters["year_preference"] == "newest":
        # Sort by modelYear descending
        return sorted(
            results,
            key=lambda x: x.payload.get("modelYear") or 0,
            reverse=True
        )
    elif filters["year_preference"] == "oldest":
        # Sort by modelYear ascending
        return sorted(
            results,
            key=lambda x: x.payload.get("modelYear") or 9999
        )
    
    return results

