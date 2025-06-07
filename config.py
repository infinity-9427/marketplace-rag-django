import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RAGConfig:
    """Configuration class for RAG system"""
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    embedding_model: str = "models/text-embedding-004"
    llm_model: str = "gemini-1.5-flash"
    temperature: float = 0.3
    max_output_tokens: int = 1000
    vector_db_path: str = "chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 15
    cache_ttl: int = 3600  # 1 hour
    max_conversation_history: int = 6
    
    def validate(self) -> bool:
        """Validate configuration"""
        return bool(self.google_api_key)

config = RAGConfig()