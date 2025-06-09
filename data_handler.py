import json
import requests
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import hashlib
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Data source configuration"""
    source_type: str  # 'file', 'api', 'database', 'supabase', 'mongodb'
    location: str  # file path, API URL, database connection string, or table name
    headers: Optional[Dict[str, str]] = None
    auth: Optional[Dict[str, str]] = None
    cache_duration: int = 3600  # seconds
    config: Optional[Dict[str, str]] = None  # For database specific config

class DatabaseAdapter(ABC):
    """Abstract base class for database adapters"""
    
    @abstractmethod
    def connect(self, config: Dict[str, str]) -> bool:
        """Connect to the database"""
        pass
    
    @abstractmethod
    def fetch_products(self, table_name: str) -> List[Dict[str, Any]]:
        """Fetch products from the database"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is active"""
        pass

class SupabaseAdapter(DatabaseAdapter):
    """Supabase database adapter"""
    
    def __init__(self):
        self.client = None
    
    def connect(self, config: Dict[str, str]) -> bool:
        """Connect to Supabase"""
        try:
            from supabase import create_client
            url = config.get('url')
            key = config.get('key')
            
            if url and key:
                self.client = create_client(url, key)
                logger.info("Supabase adapter connected successfully")
                return True
            else:
                logger.error("Supabase URL or key missing in config")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Supabase: {e}")
            return False
    
    def fetch_products(self, table_name: str) -> List[Dict[str, Any]]:
        """Fetch products from Supabase table"""
        if not self.client:
            raise Exception("Supabase client not connected")
        
        response = self.client.table(table_name).select("*").execute()
        
        if response.data:
            logger.info(f"Loaded {len(response.data)} items from Supabase table: {table_name}")
            return response.data
        else:
            logger.warning(f"No data found in Supabase table: {table_name}")
            return []
    
    def is_connected(self) -> bool:
        """Check if Supabase client is connected"""
        return self.client is not None

class MongoDBAdapter(DatabaseAdapter):
    """MongoDB database adapter"""
    
    def __init__(self):
        self.client = None
        self.db = None
    
    def connect(self, config: Dict[str, str]) -> bool:
        """Connect to MongoDB"""
        try:
            from pymongo import MongoClient
            connection_string = config.get('connection_string')
            database_name = config.get('database', 'portfolio')
            
            if connection_string:
                self.client = MongoClient(connection_string)
                self.db = self.client[database_name]
                # Test connection
                self.client.admin.command('ping')
                logger.info("MongoDB adapter connected successfully")
                return True
            else:
                logger.error("MongoDB connection string missing")
                return False
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            return False
    
    def fetch_products(self, collection_name: str) -> List[Dict[str, Any]]:
        """Fetch products from MongoDB collection"""
        if not self.db:
            raise Exception("MongoDB client not connected")
        
        collection = self.db[collection_name]
        products = list(collection.find({}))
        
        # Convert ObjectId to string for JSON serialization
        for product in products:
            if '_id' in product:
                product['id'] = str(product['_id'])
                del product['_id']
        
        logger.info(f"Loaded {len(products)} items from MongoDB collection: {collection_name}")
        return products
    
    def is_connected(self) -> bool:
        """Check if MongoDB client is connected"""
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
        except:
            pass
        return False

class DataHandler:
    """Enhanced data handler with database abstraction"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.adapters = {}
        self._initialize_adapters()
        
    def _initialize_adapters(self):
        """Initialize database adapters"""
        try:
            from config import config
            
            # Initialize Supabase adapter
            if hasattr(config, 'supabase_url') and hasattr(config, 'supabase_key'):
                supabase_adapter = SupabaseAdapter()
                supabase_config = {
                    'url': config.supabase_url,
                    'key': config.supabase_key
                }
                if supabase_adapter.connect(supabase_config):
                    self.adapters['supabase'] = supabase_adapter
            
            # Initialize MongoDB adapter if config exists
            if hasattr(config, 'mongodb_connection_string'):
                mongodb_adapter = MongoDBAdapter()
                mongodb_config = {
                    'connection_string': config.mongodb_connection_string,
                    'database': getattr(config, 'mongodb_database', 'portfolio')
                }
                if mongodb_adapter.connect(mongodb_config):
                    self.adapters['mongodb'] = mongodb_adapter
                    
        except Exception as e:
            logger.error(f"Error initializing adapters: {e}")
    
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
        """Load data from any supported source"""
        
        # Handle backward compatibility with string paths
        if isinstance(source, str):
            if source.endswith('.json'):
                source = DataSource(source_type='file', location=source)
            else:
                # Default to Supabase with table name
                source = DataSource(source_type='supabase', location=source)
        
        try:
            # Try to load from primary data source
            data = self._load_from_source(source)
            if data:
                # Normalize and cache the data
                normalized_data = self._normalize_data(data, source.source_type)
                cache_path = self._get_cache_path(f"{source.source_type}_{source.location}")
                self._save_to_cache(cache_path, normalized_data)
                return normalized_data
        except Exception as e:
            logger.error(f"Error loading from {source.source_type}: {e}")
        
        # Fallback to cache
        cache_path = self._get_cache_path(f"{source.source_type}_{source.location}")
        if self._is_cache_valid(cache_path, source.cache_duration):
            cached_data = self._load_from_cache(cache_path)
            if cached_data:
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
        elif source.source_type in self.adapters:
            adapter = self.adapters[source.source_type]
            if adapter.is_connected():
                return adapter.fetch_products(source.location)
            else:
                raise Exception(f"{source.source_type} adapter not connected")
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
        
        # Build normalized item with only fields that exist in our data
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
            
            # Preserve original for debugging
            '_original': item
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
    
    def add_adapter(self, adapter_type: str, adapter: DatabaseAdapter, config: Dict[str, str]) -> bool:
        """Add a new database adapter"""
        try:
            if adapter.connect(config):
                self.adapters[adapter_type] = adapter
                logger.info(f"Added {adapter_type} adapter successfully")
                return True
            else:
                logger.error(f"Failed to connect {adapter_type} adapter")
                return False
        except Exception as e:
            logger.error(f"Error adding {adapter_type} adapter: {e}")
            return False
    
    def get_available_adapters(self) -> List[str]:
        """Get list of available database adapters"""
        return list(self.adapters.keys())
    
    def test_connection(self, adapter_type: str) -> bool:
        """Test connection for a specific adapter"""
        if adapter_type in self.adapters:
            return self.adapters[adapter_type].is_connected()
        return False