import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    api_url: str = ''
    api_key: str = ''

class DatabaseManager:
    """Simplified database manager for Supabase only"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client = None
    
    def connect(self) -> bool:
        """Connect to Supabase"""
        try:
            from supabase import create_client
            
            if not self.config.api_url or not self.config.api_key:
                logger.error("Supabase URL and API key are required")
                return False
            
            self.client = create_client(self.config.api_url, self.config.api_key)
            
            logger.info("Supabase connection established successfully")
            return True
            
        except ImportError:
            logger.error("supabase not installed. Install with: pip install supabase")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close Supabase connection"""
        self.client = None
        logger.info("Supabase connection closed")
    
    def is_connected(self) -> bool:
        """Check if Supabase is connected"""
        return self.client is not None
    
    def fetch_products(self, table_name: str = 'products') -> List[Dict[str, Any]]:
        """Fetch all products from Supabase"""
        try:
            if not self.client:
                logger.error("Supabase client not connected")
                return []
                
            response = self.client.table(table_name).select("*").execute()
            
            products = response.data if response.data else []
            logger.info(f"Fetched {len(products)} products from Supabase")
            return products
            
        except Exception as e:
            logger.error(f"Error fetching products from Supabase: {e}")
            return []
    
    def fetch_product_by_id(self, product_id: str, table_name: str = 'products') -> Optional[Dict[str, Any]]:
        """Fetch single product by ID from Supabase"""
        try:
            if not self.client:
                logger.error("Supabase client not connected")
                return None
                
            response = self.client.table(table_name).select("*").eq('id', product_id).execute()
            
            if response.data:
                return response.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Error fetching product {product_id} from Supabase: {e}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """Supabase health check"""
        try:
            if not self.client:
                return {
                    "status": "unhealthy",
                    "database_type": "supabase",
                    "error": "Client not connected",
                    "connection_status": "disconnected"
                }
                
            response = self.client.table('products').select("id", count='exact').execute()
            
            return {
                "status": "healthy",
                "database_type": "supabase",
                "product_count": response.count,
                "connection_status": "connected"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "database_type": "supabase",
                "error": str(e),
                "connection_status": "disconnected"
            }
    
    def switch_database(self, new_config: DatabaseConfig) -> bool:
        """Switch to a different Supabase database"""
        try:
            # Disconnect current connection
            self.disconnect()
            
            # Update config and reconnect
            self.config = new_config
            return self.connect()
            
        except Exception as e:
            logger.error(f"Error switching database: {e}")
            return False

def create_database_manager() -> DatabaseManager:
    """Create database manager from environment variables or config"""
    
    # Try environment variables first
    api_url = os.getenv('SUPABASE_URL', '')
    api_key = os.getenv('SUPABASE_KEY', '')
    
    # Fallback to config file
    if not api_url or not api_key:
        try:
            from config import config as app_config
            api_url = getattr(app_config, 'supabase_url', '')
            api_key = getattr(app_config, 'supabase_key', '')
        except ImportError:
            logger.error("No config file found and environment variables not set")
    
    if not api_url or not api_key:
        raise ValueError("Supabase URL and API key are required")
    
    config = DatabaseConfig(
        api_url=api_url,
        api_key=api_key
    )
    
    return DatabaseManager(config)

# Global database manager instance
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get or create global database manager instance"""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = create_database_manager()
        _db_manager.connect()
    
    return _db_manager