"""
Canonical mappings for colors and cities.
Used during ingestion to normalize data and during query to normalize filters.
"""
import re
from typing import Dict, Optional, List

# Canonical color mappings: canonical_value -> [all_variations]
COLOR_CANON: Dict[str, List[str]] = {
    "red": [
        "red", "красный", "красная", "красный металлик",
        "бордовый", "вишневый", "алый"
    ],
    "black": [
        "black", "черный", "чёрный", "черный металлик", "черная"
    ],
    "white": [
        "white", "белый", "белый металлик", "перламутр", "белая"
    ],
    "gray": [
        "gray", "grey", "серый", "серый металлик", "графит", "серая"
    ],
    "silver": [
        "silver", "серебристый", "серебристый металлик", "серебристая"
    ],
    "blue": [
        "blue", "синий", "синий металлик", "синяя", "голубой"
    ],
    "green": [
        "green", "зеленый", "зеленый металлик", "зеленая"
    ],
    "yellow": [
        "yellow", "желтый", "жёлтый", "желтая"
    ],
    "brown": [
        "brown", "коричневый", "коричневый металлик", "коричневая"
    ],
    "bronze": [
        "bronze", "бронза"
    ],
    "beige": [
        "beige", "бежевый", "бежевая"
    ],
    "gold": [
        "gold", "золотистый", "золотистая"
    ],
    "turquoise": [
        "turquoise", "бирюзовый", "бирюзовая"
    ],
    "orange": [
        "orange", "оранжевый", "оранжевая"
    ],
    "purple": [
        "purple", "фиолетовый", "фиолетовая"
    ],
    "pink": [
        "pink", "розовый", "розовая"
    ],
}

# Reverse mapping: variation -> canonical_value
COLOR_VARIATION_TO_CANON: Dict[str, str] = {}
for canon, variations in COLOR_CANON.items():
    for variation in variations:
        COLOR_VARIATION_TO_CANON[variation.lower()] = canon

# Canonical city mappings: canonical_value -> [all_variations]
CITY_CANON: Dict[str, List[str]] = {
    "Алматы": [
        "алматы", "алма-ата", "алма ата", "алмата", "алма-ата",
        "almaty", "alma-ata", "alma ata", "Almata", "Alma-Ata",
        "алма-аты", "алма аты"
    ],
    "Астана": [
        "астана", "astana", "нур-султан", "нур султан", "нурсултан",
        "nur-sultan", "nur sultan", "nursultan", "Nur-Sultan",
        "астана-сити", "astana-city"
    ],
    "Актау": [
        "актау", "aktau", "ак-тау", "ак тау", "ak-tau", "ak tau",
        "актау-сити", "aktau-city"
    ],
    "Актобе": [
        "актобе", "aktobe", "ак-тобе", "ак тобе", "ak-tobe", "ak tobe",
        "актюбинск", "aktyubinsk", "актюбинск"
    ],
    "Атырау": [
        "атырау", "atyrau", "аты-рау", "аты рау", "aty-rau", "aty rau",
        "гурьев", "guryev", "гурьев"
    ],
    "Есик": [
        "есик", "esik", "есик-сити", "esik-city", "есик-город",
        "есик город", "esik gorod"
    ],
    "Караганда": [
        "караганда", "karaganda", "кара-ганда", "кара ганда",
        "karaganda-city", "караганда-сити", "карагандинск", "karagandinsk"
    ],
    "Каскелен": [
        "каскелен", "kaskelen", "кас-келен", "кас келен",
        "kaskelen-city", "каскелен-сити"
    ],
    "Кокшетау": [
        "кокшетау", "kokchetav", "kokchetau", "кок-шетау", "кок шетау",
        "kok-shetau", "kok shetau", "кокчетав", "kokchetav"
    ],
    "Костанай": [
        "костанай", "kostanay", "коста-най", "коста най",
        "kostanay-city", "костанай-сити", "кустанай", "kustanay"
    ],
    "Кызылорда": [
        "кызылорда", "kyzylorda", "кызыл-орда", "кызыл орда",
        "kyzyl-orda", "kyzyl orda", "кызылординск", "kyzylordinsk"
    ],
    "Павлодар": [
        "павлодар", "pavlodar", "павло-дар", "павло дар",
        "pavlo-dar", "pavlo dar", "павлодарск", "pavlodarsk"
    ],
    "Петропавловск": [
        "петропавловск", "petropavlovsk", "петро-павловск", "петро павловск",
        "petro-pavlovsk", "petro pavlovsk", "петропавловский", "petropavlovsky"
    ],
    "Семей": [
        "семей", "semey", "семеи", "сем-ей", "сем ей",
        "sem-ey", "sem ey", "семей-сити", "semey-city",
        "семеи-сити", "семеи сити"
    ],
    "Талдыкорган": [
        "талдыкорган", "taldikorgan", "талды-корган", "талды корган",
        "taldy-korgan", "taldy korgan", "талдыкорган-сити", "taldikorgan-city"
    ],
    "Тараз": [
        "тараз", "taraz", "та-раз", "та раз", "ta-raz", "ta raz",
        "тараз-сити", "taraz-city", "джамбул", "zhambyl", "джambul"
    ],
    "Туркестан": [
        "туркестан", "turkestan", "турке-стан", "турке стан",
        "turk-estan", "turk estan", "туркестан-сити", "turkestan-city"
    ],
    "Уральск": [
        "уральск", "uralsk", "ураль-ск", "ураль ск",
        "ural-sk", "ural sk", "уральск-сити", "uralsk-city"
    ],
    "Шымкент": [
        "шымкент", "shymkent", "шимкент", "shimkent", "шым-кент", "шым кент",
        "shym-kent", "shym kent", "шим-кент", "шим кент",
        "шымкент-сити", "shymkent-city", "чимкент", "chimkent"
    ],
    "Экибастуз": [
        "экибастуз", "ekibastuz", "эки-бастуз", "эки бастуз",
        "eki-bastuz", "eki bastuz", "экибастуз-сити", "ekibastuz-city"
    ],
}

# Reverse mapping: variation -> canonical_value
CITY_VARIATION_TO_CANON: Dict[str, str] = {}
for canon, variations in CITY_CANON.items():
    for variation in variations:
        CITY_VARIATION_TO_CANON[variation.lower()] = canon


def normalize_color_to_canonical(color: str) -> Optional[str]:
    """
    Normalize color to canonical value.
    
    Args:
        color: Color string (can be in any language/variation)
        
    Returns:
        Canonical color value (e.g., "red", "black", "white", "gray") or None
    """
    if not color:
        return None
    
    color_lower = color.lower().strip()
    
    # Check direct mapping
    if color_lower in COLOR_VARIATION_TO_CANON:
        return COLOR_VARIATION_TO_CANON[color_lower]
    
    # Try partial matching (e.g., "красный металлик" contains "красный")
    for variation, canon in COLOR_VARIATION_TO_CANON.items():
        if variation in color_lower or color_lower in variation:
            return canon
    
    # If no match found, return None
    return None


def normalize_city_to_canonical(city: str) -> Optional[str]:
    """
    Normalize city to canonical value.
    
    Args:
        city: City string (can be in any language/variation)
        
    Returns:
        Canonical city value (e.g., "Алматы", "Астана") or None
    """
    if not city:
        return None
    
    city_lower = city.lower().strip()
    
    # Remove common separators
    city_normalized = re.sub(r'[-\s]+', ' ', city_lower).strip()
    
    # Check direct mapping
    if city_normalized in CITY_VARIATION_TO_CANON:
        return CITY_VARIATION_TO_CANON[city_normalized]
    
    # Try without separators
    city_no_sep = re.sub(r'[-\s]+', '', city_normalized)
    if city_no_sep in CITY_VARIATION_TO_CANON:
        return CITY_VARIATION_TO_CANON[city_no_sep]
    
    # Try fuzzy matching
    for variation, canon in CITY_VARIATION_TO_CANON.items():
        variation_no_sep = re.sub(r'[-\s]+', '', variation)
        if variation_no_sep == city_no_sep:
            return canon
    
    # If no match found, return None
    return None


def normalize_filters_to_canonical(filters: Dict[str, any]) -> Dict[str, any]:
    """
    Normalize filter values to canonical format.
    Used during query time to normalize extracted filters.
    
    Args:
        filters: Dictionary of extracted filters
        
    Returns:
        Dictionary with normalized filter values (canonical colors/cities)
    """
    normalized = filters.copy()
    
    # Normalize color to canonical
    if filters.get("color"):
        canonical_color = normalize_color_to_canonical(filters["color"])
        if canonical_color:
            normalized["color"] = canonical_color
            from utils.logger import log
            if filters["color"].lower() != canonical_color.lower():
                log(f"Color normalized to canonical: '{filters['color']}' → '{canonical_color}'")
        else:
            from utils.logger import log
            log(f"Color normalization failed for: '{filters['color']}', keeping original")
    
    # Normalize city to canonical
    if filters.get("city"):
        canonical_city = normalize_city_to_canonical(filters["city"])
        if canonical_city:
            normalized["city"] = canonical_city
            from utils.logger import log
            if filters["city"].lower() != canonical_city.lower():
                log(f"City normalized to canonical: '{filters['city']}' → '{canonical_city}'")
        else:
            from utils.logger import log
            log(f"City normalization failed for: '{filters['city']}', keeping original")
    
    return normalized

