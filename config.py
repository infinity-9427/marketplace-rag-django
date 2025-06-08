import os
import socket
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Google AI settings
    google_api_key = os.getenv("GOOGLE_API_KEY")
    embedding_model = "models/text-embedding-004"
    llm_model = "gemini-2.0-flash"
    temperature = 0.1
    max_output_tokens = 1024
    
    # Pinecone settings
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "rag-ai-voice")
    
    # Supabase settings
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    # Legacy ChromaDB path (for migration reference)
    vector_db_path = "./chroma_db"
    
    def validate(self):
        """Validate configuration"""
        return bool(self.google_api_key and self.pinecone_api_key)
    
    def check_network_connectivity(self):
        """Check if we can reach external services"""
        connectivity = {
            "google": False,
            "pinecone": False,
            "supabase": False
        }
        
        # Check Google
        try:
            socket.create_connection(("generativelanguage.googleapis.com", 443), timeout=5)
            connectivity["google"] = True
        except:
            pass
        
        # Check Pinecone
        try:
            socket.create_connection(("api.pinecone.io", 443), timeout=5)
            connectivity["pinecone"] = True
        except:
            pass
        
        # Check Supabase
        if self.supabase_url:
            try:
                host = self.supabase_url.replace("https://", "").replace("http://", "")
                socket.create_connection((host, 443), timeout=5)
                connectivity["supabase"] = True
            except:
                pass
        
        return connectivity

config = Config()