import json
import requests
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import hashlib
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import the new database manager
from dbConnection import get_database_manager, DatabaseManager, DatabaseConfig

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Data source configuration"""
    source_type: str  # 'file', 'api', 'database'
    location: str  # file path, API URL, or table name
    headers: Optional[Dict[str, str]] = None
    auth: Optional[Dict[str, str]] = None
    cache_duration: int = 3600  # seconds
    config: Optional[Dict[str, str]] = None  # For database specific config

class DataHandler:
    """Enhanced data handler with optional caching"""
    
    def __init__(self, cache_dir: str = "cache", enable_cache: bool = True):
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.in_memory_cache = {}  # Use in-memory cache instead of files
        
        if enable_cache:
            os.makedirs(cache_dir, exist_ok=True)
        
        self.db_manager: Optional[DatabaseManager] = None
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            self.db_manager = get_database_manager()
            if self.db_manager.is_connected():
                logger.info("Database connection established successfully")
            else:
                logger.warning("Database connection failed, will use fallback methods")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            self.db_manager = None
    
    def _get_cache_key(self, source_key: str) -> str:
        """Generate cache key"""
        return hashlib.md5(source_key.encode()).hexdigest()
    
    def _get_cache_path(self, source_key: str) -> str:
        """Generate cache file path"""
        hash_key = self._get_cache_key(source_key)
        return os.path.join(self.cache_dir, f"data_cache_{hash_key}.json")
    
    def _is_cache_valid(self, cache_key: str, max_age: int) -> bool:
        """Check if in-memory cache is still valid"""
        if not self.enable_cache or cache_key not in self.in_memory_cache:
            return False
        
        cache_data = self.in_memory_cache[cache_key]
        cache_time = cache_data.get('timestamp', 0)
        return (datetime.now().timestamp() - cache_time) < max_age
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Load data from in-memory cache"""
        try:
            if cache_key in self.in_memory_cache:
                logger.info(f"Loaded data from memory cache")
                return self.in_memory_cache[cache_key].get('data', [])
        except Exception as e:
            logger.error(f"Error loading from memory cache: {e}")
        
        # Fallback to file cache if enabled
        if self.enable_cache:
            try:
                cache_path = self._get_cache_path(cache_key)
                if os.path.exists(cache_path):
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        logger.info(f"Loaded data from file cache: {cache_path}")
                        return cached_data.get('data', [])
            except Exception as e:
                logger.error(f"Error loading file cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: List[Dict[str, Any]]) -> None:
        """Save data to in-memory cache (and optionally file cache)"""
        try:
            # Always save to memory cache
            cache_data = {
                'timestamp': datetime.now().timestamp(),
                'data': data
            }
            self.in_memory_cache[cache_key] = cache_data
            logger.info(f"Saved data to memory cache")
            
            # Optionally save to file cache
            if self.enable_cache:
                cache_path = self._get_cache_path(cache_key)
                cache_file_data = {
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                }
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_file_data, f, indent=2)
                logger.info(f"Saved data to file cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def load_data(self, source: Union[DataSource, str]) -> List[Dict[str, Any]]:
        """Load data from any supported source"""
        
        # Handle backward compatibility with string paths
        if isinstance(source, str):
            if source.endswith('.json'):
                source = DataSource(source_type='file', location=source)
            else:
                # Default to database with table name
                source = DataSource(source_type='database', location=source)
        
        cache_key = self._get_cache_key(f"{source.source_type}_{source.location}")
        
        # Check cache first
        if self._is_cache_valid(cache_key, source.cache_duration):
            cached_data = self._load_from_cache(cache_key)
            if cached_data:
                return cached_data
        
        try:
            # Try to load from primary data source
            data = self._load_from_source(source)
            if data:
                # Normalize and cache the data
                normalized_data = self._normalize_data(data, source.source_type)
                self._save_to_cache(cache_key, normalized_data)
                return normalized_data
        except Exception as e:
            logger.error(f"Error loading from {source.source_type}: {e}")
        
        # Fallback to any cached data (even if expired)
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            logger.warning("Using expired cache data as fallback")
            return cached_data
        
        # Final fallback
        logger.warning("Using minimal fallback data")
        return self._get_minimal_fallback_data()
    
    def _load_from_source(self, source: DataSource) -> List[Dict[str, Any]]:
        """Load data from the specified source"""
        if source.source_type == 'file':
            return self._load_from_file(source.location)
        elif source.source_type == 'api':
            return self._load_from_api(source)
        elif source.source_type == 'database':
            return self._load_from_database(source.location)
        else:
            raise Exception(f"Unsupported source type: {source.source_type}")
    
    def _load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} items from file: {file_path}")
                return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []
    
    def _load_from_api(self, source: DataSource) -> List[Dict[str, Any]]:
        """Load data from API"""
        try:
            headers = source.headers or {}
            response = requests.get(source.location, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, dict) and 'products' in data:
                data = data['products']
            
            logger.info(f"Loaded {len(data)} items from API: {source.location}")
            return data
        except Exception as e:
            logger.error(f"Error loading from API {source.location}: {e}")
            return []
    
    def _load_from_database(self, table_name: str) -> List[Dict[str, Any]]:
        """Load data from database using the database manager"""
        try:
            if not self.db_manager or not self.db_manager.is_connected():
                raise Exception("Database not connected")
            
            data = self.db_manager.fetch_products(table_name)
            logger.info(f"Loaded {len(data)} items from database table: {table_name}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading from database table {table_name}: {e}")
            return []
    
    def _normalize_data(self, data: List[Dict[str, Any]], source_type: str) -> List[Dict[str, Any]]:
        """Normalize data from any source to consistent format"""
        normalized = []
        
        for item in data:
            try:
                normalized_item = self._normalize_item(item)
                if normalized_item:
                    normalized.append(normalized_item)
            except Exception as e:
                logger.warning(f"Error normalizing item {item.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Normalized {len(normalized)} items from {source_type}")
        return normalized
    
    def _normalize_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a single item to standard format"""
        
        # Extract and validate required fields
        if not item.get('name'):
            return None
        
        # Parse features from string format "feature1|feature2|feature3" or list
        features = []
        if item.get('features'):
            if isinstance(item['features'], str):
                features = [f.strip() for f in item['features'].split('|') if f.strip()]
            elif isinstance(item['features'], list):
                features = [str(f).strip() for f in item['features'] if str(f).strip()]
        
        # Parse tags from string format "tag1|tag2|tag3" or list
        tags = []
        if item.get('tags'):
            if isinstance(item['tags'], str):
                tags = [t.strip() for t in item['tags'].split('|') if t.strip()]
            elif isinstance(item['tags'], list):
                tags = [str(t).strip() for t in item['tags'] if str(t).strip()]
        
        # Parse specifications (should be JSON object)
        specifications = item.get('specifications', {})
        if isinstance(specifications, str):
            try:
                specifications = json.loads(specifications)
            except:
                specifications = {}
        
        # Build normalized item with ONLY actual data fields
        normalized = {
            'id': str(item.get('id', '')),
            'name': str(item.get('name', '')),
            'price': float(item.get('price', 0)),
            'originalPrice': float(item.get('originalPrice', item.get('price', 0))),
            'description': str(item.get('description', '')),
            'category': str(item.get('category', 'General')),
            'subcategory': str(item.get('subcategory', '')),
            'brand': str(item.get('brand', '')),
            'image': str(item.get('image', '')),
            'features': features,
            'tags': tags,
            'specifications': specifications,
            'inStock': bool(item.get('inStock', True)),
            
            # Computed fields for better RAG performance
            'price_range': self._get_price_range(float(item.get('price', 0))),
            'discount_percentage': self._calculate_discount(
                item.get('originalPrice', item.get('price', 0)), 
                item.get('price', 0)
            ),
            'search_text': self._build_search_text(item, features, tags),
        }
        
        return normalized
    
    def _get_price_range(self, price: float) -> str:
        """Categorize price into ranges"""
        if price < 50:
            return "budget"
        elif price < 200:
            return "mid-range"
        elif price < 500:
            return "premium"
        else:
            return "luxury"
    
    def _calculate_discount(self, original_price: float, current_price: float) -> float:
        """Calculate discount percentage"""
        if not original_price or original_price <= current_price:
            return 0.0
        return round(((original_price - current_price) / original_price) * 100, 1)
    
    def _build_search_text(self, item: Dict[str, Any], features: List[str], tags: List[str]) -> str:
        """Build comprehensive search text for better embeddings"""
        parts = [
            item.get('name', ''),
            item.get('description', ''),
            item.get('category', ''),
            item.get('subcategory', ''),
            item.get('brand', ''),
            ' '.join(features),
            ' '.join(tags)
        ]
        
        # Add specifications text
        specs = item.get('specifications', {})
        if isinstance(specs, dict):
            spec_text = ' '.join([f"{k} {v}" for k, v in specs.items() if v])
            parts.append(spec_text)
        
        return ' '.join([str(p) for p in parts if p]).strip()
    
    def _get_minimal_fallback_data(self) -> List[Dict[str, Any]]:
        """Minimal fallback data when all else fails"""
        return [
            {
                "id": "fallback_1",
                "name": "Sample Product",
                "price": 99.99,
                "originalPrice": 119.99,
                "description": "Sample product for testing purposes",
                "category": "Electronics",
                "subcategory": "Gadgets",
                "brand": "SampleBrand",
                "image": "",
                "features": ["Sample Feature"],
                "tags": ["sample"],
                "specifications": {"type": "sample"},
                "inStock": True,
                "price_range": "mid-range",
                "discount_percentage": 16.7,
                "search_text": "Sample Product Electronics Gadgets SampleBrand Sample Feature sample type sample"
            }
        ]
    
    def get_categories(self, data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract all categories and subcategories dynamically"""
        categories = {}
        
        for item in data:
            category = item.get('category', 'General')
            subcategory = item.get('subcategory', '')
            
            if category not in categories:
                categories[category] = set()
            
            if subcategory:
                categories[category].add(subcategory)
        
        # Convert sets to sorted lists
        return {cat: sorted(list(subcats)) for cat, subcats in categories.items()}
    
    def get_brands(self, data: List[Dict[str, Any]]) -> List[str]:
        """Extract all brands dynamically"""
        brands = set()
        for item in data:
            brand = item.get('brand', '').strip()
            if brand:
                brands.add(brand)
        return sorted(list(brands))
    
    def get_price_ranges(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get price statistics"""
        prices = [item.get('price', 0) for item in data if item.get('price', 0) > 0]
        
        if not prices:
            return {}
        
        sorted_prices = sorted(prices)
        median = sorted_prices[len(sorted_prices) // 2] if len(sorted_prices) % 2 == 1 else (sorted_prices[len(sorted_prices) // 2 - 1] + sorted_prices[len(sorted_prices) // 2]) / 2
        
        return {
            "min": min(prices),
            "max": max(prices),
            "avg": sum(prices) / len(prices),
            "median": median
        }
    
    def switch_database(self, new_api_url: str, new_api_key: str) -> bool:
        """Switch to a different Supabase database"""
        try:
            if self.db_manager:
                new_config = DatabaseConfig(api_url=new_api_url, api_key=new_api_key)
                return self.db_manager.switch_database(new_config)
            return False
        except Exception as e:
            logger.error(f"Error switching database: {e}")
            return False
    
    def get_database_health(self) -> Dict[str, Any]:
        """Get database health status"""
        if self.db_manager:
            return self.db_manager.health_check()
        return {
            "status": "unhealthy",
            "error": "No database manager initialized"
        }
    
    def get_available_databases(self) -> List[str]:
        """Get list of supported database types"""
        return ['supabase']