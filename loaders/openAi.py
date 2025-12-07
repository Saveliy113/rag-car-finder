import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional

# Load environment variables
load_dotenv()

# Singleton instance
_openai_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client singleton"""
    global _openai_client
    
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        _openai_client = OpenAI(api_key=api_key)
    
    return _openai_client


def init_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Initialize OpenAI client (called at startup)"""
    global _openai_client
    
    if _openai_client is None:
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
        _openai_client = OpenAI(api_key=api_key)
    
    return _openai_client