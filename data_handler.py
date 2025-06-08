import json
import requests
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import hashlib
import os
from dataclasses import dataclass
from supabase import create_client, Client

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Data source configuration"""
    source_type: str  # 'file', 'api', 'database', 'supabase'
    location: str  # file path, API URL, database connection string, or table name for Supabase
    headers: Optional[Dict[str, str]] = None
    auth: Optional[Dict[str, str]] = None
    cache_duration: int = 3600  # seconds
    supabase_config: Optional[Dict[str, str]] = None  # For Supabase specific config

class DataHandler:
    """Enhanced data handler supporting multiple data sources including Supabase"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.supabase_client = None
        self._initialize_supabase()
        
    def _initialize_supabase(self):
        """Initialize Supabase client"""
        try:
            from config import config
            if config.supabase_url and config.supabase_key:
                self.supabase_client = create_client(config.supabase_url, config.supabase_key)
                logger.info("Supabase client initialized successfully")
            else:
                logger.warning("Supabase credentials not found in config")
        except Exception as e:
            logger.error(f"Error initializing Supabase client: {e}")
    
    def _get_cache_path(self, source_key: str) -> str:
        """Generate cache file path"""
        hash_key = hashlib.md5(source_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"data_cache_{hash_key}.json")
    
    def _is_cache_valid(self, cache_path: str, max_age: int) -> bool:
        """Check if cache is still valid"""
        if not os.path.exists(cache_path):
            return False
        
        cache_time = os.path.getmtime(cache_path)
        return (datetime.now().timestamp() - cache_time) < max_age
    
    def _load_from_cache(self, cache_path: str) -> Optional[List[Dict[str, Any]]]:
        """Load data from cache"""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                logger.info(f"Loaded data from cache: {cache_path}")
                return cached_data.get('data', [])
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None
    
    def _save_to_cache(self, cache_path: str, data: List[Dict[str, Any]]) -> None:
        """Save data to cache"""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Saved data to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def load_data(self, source: Union[DataSource, str]) -> List[Dict[str, Any]]:
        """Load data from various sources with caching"""
        
        # Handle backward compatibility with string paths
        if isinstance(source, str):
            source = DataSource(source_type='file', location=source)
        
        cache_path = self._get_cache_path(source.location)
        
        # Check cache first (except for Supabase which should be more real-time)
        if source.source_type != 'supabase' and self._is_cache_valid(cache_path, source.cache_duration):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        # Load fresh data
        try:
            if source.source_type == 'file':
                data = self._load_from_file(source.location)
            elif source.source_type == 'api':
                data = self._load_from_api(source)
            elif source.source_type == 'supabase':
                data = self._load_from_supabase(source)
            else:
                raise ValueError(f"Unsupported source type: {source.source_type}")
            
            # Validate and normalize data
            normalized_data = self._normalize_data(data)
            
            # Cache the data
            self._save_to_cache(cache_path, normalized_data)
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Error loading data from {source.location}: {e}")
            # Try to return stale cache as fallback
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                logger.warning("Using stale cache data as fallback")
                return cached_data
            raise
    
    def _load_from_supabase(self, source: DataSource) -> List[Dict[str, Any]]:
        """Load data from Supabase table with fallback"""
        if not self.supabase_client:
            logger.warning("Supabase client not initialized, falling back to local data")
            return self._load_fallback_data()
        
        try:
            # Use the location as table name (default: "products")
            table_name = source.location or "products"
            
            # Query the table with timeout
            response = self.supabase_client.table(table_name).select("*").execute()
            
            if response.data:
                data = response.data
                logger.info(f"Loaded {len(data)} items from Supabase table: {table_name}")
                return data
            else:
                logger.warning(f"No data found in Supabase table: {table_name}")
                return self._load_fallback_data()
                
        except Exception as e:
            logger.error(f"Error loading from Supabase: {e}")
            logger.info("Attempting to use fallback data source")
            return self._load_fallback_data()
    
    def _load_fallback_data(self) -> List[Dict[str, Any]]:
        """Load fallback data when Supabase is unavailable"""
        fallback_paths = [
            "assistant/products.json",
            "products.json",
            "sample_products.json"
        ]
        
        for path in fallback_paths:
            try:
                if os.path.exists(path):
                    logger.info(f"Using fallback data from: {path}")
                    return self._load_from_file(path)
            except Exception as e:
                logger.warning(f"Could not load fallback file {path}: {e}")
                continue
        
        # Return minimal sample data if no files available
        logger.warning("No fallback data available, using minimal sample")
        return [
            {
                "id": "sample_1",
                "name": "Sample Wireless Headphones",
                "price": 99.99,
                "description": "High-quality wireless headphones with noise cancellation",
                "category": "Audio",
                "in_stock": True,
                "features": ["Wireless", "Noise Cancellation", "Long Battery Life"],
                "rating": 4.5,
                "brand": "SampleBrand"
            }
        ]
    
    def _load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} items from file: {file_path}")
        return data
    
    def _load_from_api(self, source: DataSource) -> List[Dict[str, Any]]:
        """Load data from API endpoint"""
        headers = source.headers or {}
        
        # Add authentication if provided
        if source.auth:
            if source.auth.get('type') == 'bearer':
                headers['Authorization'] = f"Bearer {source.auth.get('token')}"
            elif source.auth.get('type') == 'api_key':
                headers[source.auth.get('header', 'X-API-Key')] = source.auth.get('key')
        
        response = requests.get(source.location, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle different API response formats
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            elif 'items' in data:
                data = data['items']
            elif 'products' in data:
                data = data['products']
            elif 'results' in data:
                data = data['results']
        
        if not isinstance(data, list):
            raise ValueError("API response is not a list or doesn't contain a list")
        
        logger.info(f"Loaded {len(data)} items from API: {source.location}")
        return data
    
    def _normalize_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize data structure to ensure consistent format"""
        normalized = []
        
        for item in data:
            try:
                normalized_item = self._normalize_item(item)
                if normalized_item:
                    normalized.append(normalized_item)
            except Exception as e:
                logger.warning(f"Error normalizing item: {e}")
                continue
        
        logger.info(f"Normalized {len(normalized)} items")
        return normalized
    
    def _normalize_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a single item to standard format"""
        
        # Required fields mapping (flexible field names)
        name_fields = ['name', 'title', 'product_name', 'item_name']
        price_fields = ['price', 'cost', 'amount', 'value']
        description_fields = ['description', 'desc', 'summary', 'details']
        category_fields = ['category', 'type', 'group', 'class']
        
        # Extract name (required)
        name = None
        for field in name_fields:
            if field in item and item[field]:
                name = str(item[field]).strip()
                break
        
        if not name:
            logger.warning("Item missing name field")
            return None
        
        # Extract other fields with fallbacks
        normalized = {
            'name': name,
            'price': self._extract_price(item, price_fields),
            'description': self._extract_text_field(item, description_fields),
            'category': self._extract_text_field(item, category_fields) or 'general',
            'in_stock': self._extract_boolean_field(item, ['in_stock', 'inStock', 'available']),
            'features': self._extract_features(item),
            'rating': self._extract_numeric_field(item, ['rating', 'score'], 0),
            'reviews': self._extract_numeric_field(item, ['reviews', 'review_count'], 0),
            'id': str(item.get('id', item.get('product_id', name.lower().replace(' ', '_')))),
            'brand': self._extract_text_field(item, ['brand', 'manufacturer']),
            'subcategory': self._extract_text_field(item, ['subcategory', 'subCategory']),
            'image': self._extract_text_field(item, ['image', 'imageUrl', 'image_url']),
        }
        
        # Preserve original data
        normalized['_original'] = item
        
        return normalized
    
    def _extract_price(self, item: Dict[str, Any], price_fields: List[str]) -> float:
        """Extract and normalize price"""
        for field in price_fields:
            if field in item:
                try:
                    price_value = item[field]
                    if isinstance(price_value, str):
                        # Remove currency symbols and convert
                        price_value = price_value.replace('$', '').replace(',', '').strip()
                    return float(price_value)
                except (ValueError, TypeError):
                    continue
        return 0.0
    
    def _extract_text_field(self, item: Dict[str, Any], fields: List[str]) -> str:
        """Extract text field with fallbacks"""
        for field in fields:
            if field in item and item[field]:
                return str(item[field]).strip()
        return ""
    
    def _extract_boolean_field(self, item: Dict[str, Any], fields: List[str]) -> bool:
        """Extract boolean field with fallbacks"""
        for field in fields:
            if field in item:
                value = item[field]
                if isinstance(value, bool):
                    return value
                elif isinstance(value, str):
                    return value.lower() in ['true', '1', 'yes', 'on']
                elif isinstance(value, (int, float)):
                    return bool(value)
        return True  # Default to in stock
    
    def _extract_numeric_field(self, item: Dict[str, Any], fields: List[str], default: float = 0.0) -> float:
        """Extract numeric field with fallbacks"""
        for field in fields:
            if field in item:
                try:
                    return float(item[field])
                except (ValueError, TypeError):
                    continue
        return default
    
    def _extract_features(self, item: Dict[str, Any]) -> List[str]:
        """Extract features list"""
        features_fields = ['features', 'specs', 'specifications', 'attributes']
        
        for field in features_fields:
            if field in item:
                features = item[field]
                if isinstance(features, list):
                    return [str(f).strip() for f in features if f]
                elif isinstance(features, str):
                    # Try to split on common delimiters
                    return [f.strip() for f in features.split(',') if f.strip()]
        
        return []