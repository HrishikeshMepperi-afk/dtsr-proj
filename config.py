import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration settings."""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Vector DB settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # LLM Settings
    LLM_MODEL = "llama-3.1-8b-instant"
    
    @classmethod
    def validate(cls):
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is not set. Please check your .env file.")

# Validate config on import to catch issues early
Config.validate()
