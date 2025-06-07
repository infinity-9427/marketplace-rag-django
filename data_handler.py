import json
import requests
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import hashlib
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Data source configuration"""
    source_type: str  # 'file', 'api', 'database'
    location: str  # file path, API URL, or database connection string
    headers: Optional[Dict[str, str]] = None
    auth: Optional[Dict[str, str]] = None
    cache_duration: int = 3600  # seconds

class DataHandler:
    """Enhanced data handler supporting multiple data sources"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
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
        
        # Check cache first
        if self._is_cache_valid(cache_path, source.cache_duration):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        # Load fresh data
        try:
            if source.source_type == 'file':
                data = self._load_from_file(source.location)
            elif source.source_type == 'api':
                data = self._load_from_api(source)
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
            'in_stock': item.get('in_stock', item.get('inStock', item.get('available', True))),
            'features': self._extract_features(item),
            'rating': item.get('rating', item.get('score', 0)),
            'reviews': item.get('reviews', item.get('review_count', 0)),
            'id': str(item.get('id', item.get('product_id', name.lower().replace(' ', '_')))),
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