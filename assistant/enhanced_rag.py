import os
import json
import logging
import traceback
import re
import time
import numpy as np
import hashlib
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Updated imports for Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from config import config
from data_handler import DataHandler, DataSource

# Configure in-memory logging only (no file generation)
logger = logging.getLogger(__name__)

# Global singleton management
_rag_system = None
_rag_system_lock = False

class ProductSimilarityEngine:
    """Engine for finding similar products based on embeddings"""
    
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.product_embeddings = {}
        self.product_data = {}
        self.similarity_cache = {}
    
    def update_products(self, products_data: List[Dict[str, Any]]):
        """Update product data and compute embeddings"""
        self.product_data = {product['id']: product for product in products_data}
        self._compute_product_embeddings(products_data)
    
    def _compute_product_embeddings(self, products_data: List[Dict[str, Any]]):
        """Compute embeddings for all products"""
        logger.info("Computing product embeddings for similarity...")
        
        for product in products_data:
            try:
                # Create comprehensive product description for embedding
                product_text = self._create_embedding_text(product)
                
                # Get embedding
                embedding = self.embeddings_model.embed_query(product_text)
                self.product_embeddings[product['id']] = embedding
                
            except Exception as e:
                logger.warning(f"Error computing embedding for product {product['id']}: {e}")
        
        logger.info(f"Computed embeddings for {len(self.product_embeddings)} products")
    
    def _create_embedding_text(self, product: Dict[str, Any]) -> str:
        """Create optimized text for embedding computation"""
        parts = [
            product['name'],
            product['description'],
            product['category'],
            f"price {product['price']}",
        ]
        
        # Add features
        if product.get('features'):
            parts.extend(product['features'])
        
        # Add price range category
        price = product['price']
        if price < 50:
            parts.append("budget affordable cheap")
        elif price < 150:
            parts.append("mid-range moderate")
        elif price < 300:
            parts.append("premium quality")
        else:
            parts.append("luxury high-end expensive")
        
        return ' '.join(parts)
    
    def find_similar_products(self, product_id: str, top_k: int = 3, 
                            same_category_only: bool = False) -> List[Dict[str, Any]]:
        """Find similar products to a given product"""
        if product_id not in self.product_embeddings:
            return []
        
        target_embedding = self.product_embeddings[product_id]
        target_product = self.product_data[product_id]
        similarities = []
        
        for other_id, other_embedding in self.product_embeddings.items():
            if other_id == product_id:
                continue
            
            other_product = self.product_data[other_id]
            
            # Filter by category if requested
            if same_category_only and other_product['category'] != target_product['category']:
                continue
            
            # Skip out of stock products
            if not other_product.get('inStock', True):
                continue
            
            # Compute similarity
            similarity = cosine_similarity([target_embedding], [other_embedding])[0][0]
            
            # Add contextual similarity factors
            context_score = self._compute_context_similarity(target_product, other_product)
            final_score = similarity * 0.7 + context_score * 0.3
            
            similarities.append({
                'product_id': other_id,
                'product': other_product,
                'similarity_score': float(final_score),
                'embedding_similarity': float(similarity),
                'context_similarity': float(context_score)
            })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:top_k]
    
    def _compute_context_similarity(self, product1: Dict[str, Any], 
                                  product2: Dict[str, Any]) -> float:
        """Compute contextual similarity beyond embeddings"""
        score = 0.0
        
        # Price similarity (inverse of price difference, normalized)
        price_diff = abs(product1['price'] - product2['price'])
        max_price = max(product1['price'], product2['price'])
        if max_price > 0:
            price_similarity = 1 - (price_diff / max_price)
            score += price_similarity * 0.5
        
        # Category match
        if product1['category'] == product2['category']:
            score += 0.4
        
        # Brand similarity
        if product1.get('brand') == product2.get('brand') and product1.get('brand'):
            score += 0.1
        
        return min(score, 1.0)

    def get_recommendation_reason(self, target_product: Dict[str, Any], 
                                similar_product: Dict[str, Any]) -> str:
        """Generate reason for recommendation"""
        reasons = []
        
        # Price comparison
        price_diff = similar_product['price'] - target_product['price']
        if abs(price_diff) > 10:
            if price_diff < 0:
                reasons.append(f"${abs(price_diff):.0f} less expensive")
            else:
                reasons.append(f"${price_diff:.0f} more but with additional features")
        else:
            reasons.append("similar price point")
        
        # Feature comparison
        target_features = set(target_product.get('features', []))
        similar_features = set(similar_product.get('features', []))
        
        if similar_features - target_features:
            reasons.append("additional features")
        
        # Category difference
        if target_product['category'] != similar_product['category']:
            reasons.append(f"alternative in {similar_product['category']}")
        
        # Brand comparison
        if (target_product.get('brand') != similar_product.get('brand') and 
            similar_product.get('brand')):
            reasons.append(f"from {similar_product['brand']}")
        
        if reasons:
            return f"offers {', '.join(reasons)}"
        else:
            return "similar specifications and quality"

class PineconeDataManager:
    """Manages Pinecone data synchronization with Supabase with strict data validation"""
    
    def __init__(self, pinecone_client, index, embeddings_model, data_handler):
        self.pc = pinecone_client
        self.index = index
        self.embeddings = embeddings_model
        self.data_handler = data_handler
        self.last_refresh_time = None
        self.refresh_interval = 300  # 5 minutes in seconds
        self.data_hash = None
        self.expected_product_count = 0
        self.documents_per_product = 1  # Only 1 comprehensive document per product
        
        # Initialize data hash on startup
        try:
            # Try to load existing data to establish baseline hash
            data_source = DataSource(source_type='supabase', location='products', cache_duration=0)
            current_data = self.data_handler.load_data(data_source)
            if current_data:
                self.data_hash = self.get_data_hash(current_data)
                logger.info(f"📊 Initialized data hash for {len(current_data)} products")
        except Exception as e:
            logger.warning(f"Could not initialize data hash: {e}")
    
    def should_refresh(self) -> bool:
        """Check if data should be refreshed based on time and data changes"""
        if self.last_refresh_time is None:
            return True
        
        time_since_refresh = time.time() - self.last_refresh_time
        return time_since_refresh > self.refresh_interval
    
    def get_data_hash(self, products_data: List[Dict[str, Any]]) -> str:
        """Generate hash of current data for change detection"""
        data_str = json.dumps(products_data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def has_data_changed(self, products_data: List[Dict[str, Any]]) -> bool:
        """Check if data has changed since last sync"""
        current_hash = self.get_data_hash(products_data)
        if self.data_hash != current_hash:
            self.data_hash = current_hash
            return True
        return False
    
    def validate_data_consistency(self, products_data: List[Dict[str, Any]]) -> bool:
        """Validate that Pinecone data is consistent with source data - 1:1 mapping"""
        try:
            stats = self.index.describe_index_stats()
            current_vector_count = stats.get('total_vector_count', 0)
            expected_vectors = len(products_data)  # FIXED: Exactly 1 vector per product
            
            logger.info(f"Data validation: {current_vector_count} vectors vs expected {expected_vectors}")
            
            # Allow small variance (±2) for timing issues, but not major discrepancies
            if abs(current_vector_count - expected_vectors) > 2:
                logger.warning(f"Data inconsistency detected: {current_vector_count} != {expected_vectors}. Forcing rebuild.")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data consistency: {e}")
            return False
    
    def refresh_pinecone_data(self, force_refresh: bool = False) -> bool:
        """Refresh Pinecone with latest Supabase data - ONE DOCUMENT PER PRODUCT"""
        try:
            logger.info("🔍 Checking for data refresh...")
            
            # Load fresh data from Supabase
            data_source = DataSource(
                source_type='supabase', 
                location='products',
                cache_duration=0  # No cache for fresh data
            )
            
            fresh_products_data = self.data_handler.load_data(data_source)
            
            if not fresh_products_data:
                logger.warning("❌ No fresh data available from Supabase")
                return False
            
            # Validate data consistency
            data_consistent = self.validate_data_consistency(fresh_products_data)
            
            # Check if refresh is needed
            if (not force_refresh and 
                not self.should_refresh() and 
                not self.has_data_changed(fresh_products_data) and 
                data_consistent):
                logger.debug("✅ Data is up to date and consistent, skipping refresh")
                return True
            
            logger.info(f"🔄 Refreshing Pinecone with {len(fresh_products_data)} fresh products...")
            self.expected_product_count = len(fresh_products_data)
            
            # Use incremental update strategy to prevent vicious cycles
            return self._incremental_pinecone_update(fresh_products_data)
                
        except Exception as e:
            logger.error(f"❌ Error refreshing Pinecone data: {e}")
            traceback.print_exc()
            return False
    
    def _incremental_pinecone_update(self, products_data: List[Dict[str, Any]]) -> bool:
        """Incremental update strategy to prevent vicious add/delete cycles"""
        try:
            logger.info("🔄 Starting incremental Pinecone update...")
            
            # Get current vectors in Pinecone
            current_vectors = self._get_current_vector_ids()
            
            # Create new documents (ONE per product)
            new_documents = self._create_comprehensive_documents(products_data)
            expected_ids = set(f"product_{product['id']}" for product in products_data)
            
            # Determine what needs to be updated
            vectors_to_delete = current_vectors - expected_ids
            vectors_to_add = expected_ids - current_vectors
            vectors_to_update = current_vectors & expected_ids
            
            logger.info(f"📊 Update plan: Delete {len(vectors_to_delete)}, Add {len(vectors_to_add)}, Update {len(vectors_to_update)}")
            
            # Step 1: Delete obsolete vectors
            if vectors_to_delete:
                logger.info(f"🗑️ Deleting {len(vectors_to_delete)} obsolete vectors...")
                try:
                    self.index.delete(ids=list(vectors_to_delete))
                    time.sleep(2)  # Wait for deletion
                    logger.info(f"✅ Deleted {len(vectors_to_delete)} obsolete vectors")
                except Exception as e:
                    logger.error(f"❌ Error deleting vectors: {e}")
            
            # Step 2: Add new vectors
            new_docs_to_add = [doc for doc in new_documents 
                              if f"product_{doc.metadata['product_id']}" in vectors_to_add]
            
            if new_docs_to_add:
                logger.info(f"➕ Adding {len(new_docs_to_add)} new vectors...")
                success = self._batch_add_documents(new_docs_to_add, "add")
                if not success:
                    return False
            
            # Step 3: Update existing vectors (upsert)
            docs_to_update = [doc for doc in new_documents 
                             if f"product_{doc.metadata['product_id']}" in vectors_to_update]
            
            if docs_to_update:
                logger.info(f"🔄 Updating {len(docs_to_update)} existing vectors...")
                success = self._batch_add_documents(docs_to_update, "update")
                if not success:
                    return False
            
            # Step 4: Verify final state
            time.sleep(3)  # Wait for all operations to complete
            final_stats = self.index.describe_index_stats()
            final_count = final_stats.get('total_vector_count', 0)
            expected_count = len(products_data)
            
            if abs(final_count - expected_count) > 1:  # Allow small variance
                logger.error(f"❌ Final validation failed: {final_count} vectors != {expected_count} products")
                return False
            
            logger.info(f"✅ Incremental update completed: {final_count} vectors for {expected_count} products")
            
            # Update tracking data
            self.last_refresh_time = time.time()
            self.data_hash = self.get_data_hash(products_data)
            
            return True
                
        except Exception as e:
            logger.error(f"❌ Error in incremental update: {e}")
            traceback.print_exc()
            return False
    
    def _get_current_vector_ids(self) -> set:
        """Get all current vector IDs in Pinecone"""
        try:
            # Query with dummy vector to get all IDs
            dummy_vector = [0.0] * 768  # Match embedding dimension
            
            # Fetch all vectors in batches
            all_ids = set()
            
            # Use list operation to get vector IDs
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            if total_vectors == 0:
                return set()
            
            # Since Pinecone doesn't have a direct "list all IDs" method,
            # we'll use a different approach: query with very broad filter
            try:
                # Query for vectors with metadata filter that should match all
                query_result = self.index.query(
                    vector=dummy_vector,
                    top_k=min(total_vectors, 10000),  # Max limit
                    include_metadata=True
                )
                
                current_ids = set()
                for match in query_result.get('matches', []):
                    current_ids.add(match['id'])
                
                logger.info(f"📊 Found {len(current_ids)} existing vectors in Pinecone")
                return current_ids
                
            except Exception as e:
                logger.warning(f"Could not query existing vectors: {e}")
                # Fallback: assume no existing vectors (will do full refresh)
                return set()
                
        except Exception as e:
            logger.error(f"Error getting current vector IDs: {e}")
            return set()
    
    def _batch_add_documents(self, documents: List[Document], operation_type: str) -> bool:
        """Add documents to Pinecone in batches"""
        try:
            vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                text_key="page_content"
            )
            
            batch_size = 10  # Smaller batches for reliability
            total_processed = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                try:
                    # Use consistent IDs based on product ID
                    document_ids = [f"product_{doc.metadata['product_id']}" for doc in batch]
                    
                    # Add/Update documents with explicit IDs
                    vectorstore.add_documents(documents=batch, ids=document_ids)
                    total_processed += len(batch)
                    
                    logger.info(f"📤 {operation_type.title()} batch {batch_num}: {len(batch)} documents (Total: {total_processed})")
                    
                    # Wait between batches
                    if i + batch_size < len(documents):
                        time.sleep(1)
                        
                except Exception as batch_error:
                    logger.error(f"❌ Error in {operation_type} batch {batch_num}: {batch_error}")
                    continue
            
            logger.info(f"✅ {operation_type.title()} completed: {total_processed} documents processed")
            return total_processed == len(documents)
            
        except Exception as e:
            logger.error(f"❌ Error in batch {operation_type}: {e}")
            return False
    
    def _create_comprehensive_documents(self, products_data: List[Dict[str, Any]]) -> List[Document]:
        """Create ONE comprehensive document per product - NO multiple documents"""
        documents = []
        current_timestamp = datetime.now().isoformat()
        
        for product in products_data:
            try:
                # Create ONE comprehensive document that includes EVERYTHING
                content = self._create_comprehensive_product_content(product)
                
                # Complete metadata
                metadata = {
                    'product_id': str(product['id']),
                    'product_name': product['name'],
                    'category': product['category'],
                    'subcategory': product.get('subcategory', ''),
                    'brand': product.get('brand', ''),
                    'price': float(product['price']),
                    'original_price': float(product.get('originalPrice', product['price'])),
                    'in_stock': bool(product.get('inStock', True)),
                    'price_range': product.get('price_range', 'mid-range'),
                    'last_updated': current_timestamp,
                    'features_count': len(product.get('features', [])),
                    'has_discount': product.get('originalPrice', 0) > product.get('price', 0)
                }
                
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
                
            except Exception as e:
                logger.warning(f"❌ Error creating document for product {product.get('id')}: {e}")
                continue
        
        logger.info(f"📝 Created {len(documents)} comprehensive documents for {len(products_data)} products (1:1 ratio)")
        return documents
    
    def _create_comprehensive_product_content(self, product: Dict[str, Any]) -> str:
        """Create ONE comprehensive document that includes ALL product information"""
        content_parts = [
            f"Product: {product['name']}",
            f"Category: {product['category']}",
            f"Price: ${product['price']:.2f}",
            f"Description: {product['description']}"
        ]
        
        if product.get('brand'):
            content_parts.append(f"Brand: {product['brand']}")
        
        if product.get('subcategory'):
            content_parts.append(f"Subcategory: {product['subcategory']}")
        
        # Include ALL features in the main document
        if product.get('features'):
            features_text = ", ".join(product['features'])
            content_parts.append(f"Key Features: {features_text}")
            
            # Add individual feature details for better searchability
            for i, feature in enumerate(product['features'], 1):
                content_parts.append(f"Feature {i}: {feature}")
        
        if product.get('tags'):
            tags_text = ", ".join(product['tags'])
            content_parts.append(f"Tags: {tags_text}")
        
        # Add availability and pricing info
        stock_status = "In Stock" if product.get('inStock', True) else "Out of Stock"
        content_parts.append(f"Availability: {stock_status}")
        
        if product.get('originalPrice') and product['originalPrice'] > product['price']:
            discount = ((product['originalPrice'] - product['price']) / product['originalPrice']) * 100
            content_parts.append(f"Discounted: {discount:.1f}% off from ${product['originalPrice']:.2f}")
        
        # Add specifications if available
        if product.get('specifications'):
            specs = product['specifications']
            if isinstance(specs, dict):
                spec_text = " ".join([f"{k}: {v}" for k, v in specs.items() if v])
                content_parts.append(f"Specifications: {spec_text}")
        
        # Add search-optimized content
        price_range = product.get('price_range', 'mid-range')
        content_parts.append(f"Price Range: {price_range}")
        
        # Add semantic search terms based on price
        price = product['price']
        if price < 50:
            content_parts.append("Budget-friendly affordable inexpensive cheap low-cost")
        elif price < 150:
            content_parts.append("Mid-range moderate reasonably-priced good-value")
        elif price < 300:
            content_parts.append("Premium quality high-end professional")
        else:
            content_parts.append("Luxury expensive high-end premium top-tier")
        
        return "\n".join(content_parts)

class PerformanceMonitor:
    """Monitor RAG system performance"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def start_timer(self, operation: str) -> float:
        """Start timing an operation"""
        start_time = time.time()
        logger.debug(f"Started {operation}")
        return start_time
    
    def end_timer(self, operation: str, start_time: float) -> float:
        """End timing and record metric"""
        duration = time.time() - start_time
        self.metrics[operation].append(duration)
        logger.debug(f"Completed {operation} in {duration:.3f}s")
        return duration
    
    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation"""
        times = self.metrics.get(operation, [])
        return sum(times) / len(times) if times else 0.0
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics"""
        stats = {}
        for operation, times in self.metrics.items():
            stats[operation] = {
                'count': len(times),
                'average': sum(times) / len(times),
                'total': sum(times),
                'min': min(times),
                'max': max(times)
            }
        return stats

class EnhancedRAGSystem:
    """Enhanced RAG System with Pinecone vector store and automatic data refresh"""
    
    def __init__(self, data_source: Optional[Union[DataSource, str]] = None):
        logger.info("🚀 Initializing Enhanced RAG System...")
        
        self.data_handler = DataHandler(enable_cache=False)
        self.performance_monitor = PerformanceMonitor()
        self.vectorstore = None
        self.qa_chain = None
        self.conversation_history = []
        self.dynamic_keywords = {}
        self.category_keywords = {}
        self.products_data = []
        self.categories = {}
        self.brands = []
        self.is_initialized = False
        
        # Add phrase tracking for variety
        self.used_opening_phrases = []
        self.used_transition_phrases = []
        self.phrase_rotation_limit = 5  # Remember last 5 phrases to avoid repetition
        
        # Initialize Pinecone client
        self.pc = None
        self.index = None
        self.pinecone_manager = None
        
        # Set default data source to Supabase
        if data_source is None:
            data_source = DataSource(
                source_type='supabase', 
                location='products',
                cache_duration=0  # Always fetch fresh data
            )
        elif isinstance(data_source, str):
            if data_source.endswith('.json'):
                data_source = DataSource(source_type='file', location=data_source)
            else:
                data_source = DataSource(source_type='supabase', location=data_source, cache_duration=0)
        
        self.data_source = data_source
        
        # Initialize models and Pinecone
        self._initialize_models()
        self._initialize_pinecone()
        self.similarity_engine = ProductSimilarityEngine(self.embeddings)
        
        # IMMEDIATE DATA LOADING AND INDEXING ON STARTUP
        logger.info("🔄 Loading and indexing data immediately on startup...")
        startup_success = self.initial_data_load()
        if startup_success:
            logger.info("✅ Initial data loading and indexing completed successfully")
            self.is_initialized = True
        else:
            logger.warning("⚠️ Initial data loading failed, will retry on first query")
    
    def initial_data_load(self) -> bool:
        """Load and index data immediately on startup with validation"""
        try:
            start_time = time.time()
            logger.info("🔄 Starting initial data load from Supabase...")
            
            # Load fresh products data
            fresh_products_data = self.data_handler.load_data(self.data_source)
            
            if not fresh_products_data:
                logger.error("❌ No data available from Supabase on startup")
                return False
            
            logger.info(f"📊 Loaded {len(fresh_products_data)} products from Supabase")
            
            # Update local data immediately
            self.products_data = fresh_products_data
            self.categories = self.data_handler.get_categories(fresh_products_data)
            self.brands = self.data_handler.get_brands(fresh_products_data)
            
            # Update similarity engine
            self.similarity_engine.update_products(fresh_products_data)
            
            # IMMEDIATE PINECONE INDEXING with validation
            if self.pinecone_manager:
                logger.info("🔧 Starting immediate Pinecone indexing...")
                indexing_success = self._immediate_pinecone_index(fresh_products_data)
                
                if indexing_success:
                    # Verify indexing with strict validation
                    time.sleep(3)  # Wait for indexing to complete
                    stats = self.index.describe_index_stats()
                    vector_count = stats.get('total_vector_count', 0)
                    
                    # Validate vector count is reasonable
                    max_expected = len(fresh_products_data) * 4  # 1 main + 3 feature docs max
                    
                    if vector_count > 0 and vector_count <= max_expected:
                        logger.info(f"✅ Pinecone indexing successful: {vector_count} vectors indexed for {len(fresh_products_data)} products")
                        self.pinecone_manager.last_refresh_time = time.time()
                        self.pinecone_manager.data_hash = self.pinecone_manager.get_data_hash(fresh_products_data)
                    else:
                        logger.warning(f"⚠️ Vector count validation failed: {vector_count} vectors for {len(fresh_products_data)} products (max expected: {max_expected})")
                        return False
                else:
                    logger.error("❌ Pinecone indexing failed")
                    return False
            
            total_time = time.time() - start_time
            logger.info(f"🎉 Initial data load completed in {total_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in initial data load: {e}")
            traceback.print_exc()
            return False
    
    def _immediate_pinecone_index(self, products_data: List[Dict[str, Any]]) -> bool:
        """Immediately index data in Pinecone on startup - ONE DOCUMENT PER PRODUCT"""
        try:
            logger.info("🗂️ Creating comprehensive documents for immediate indexing...")
            
            # Create ONE comprehensive document per product
            documents = self.pinecone_manager._create_comprehensive_documents(products_data)
            
            if not documents:
                logger.error("❌ No documents created for indexing")
                return False
            
            # Validate document count - should be exactly 1:1
            if len(documents) != len(products_data):
                logger.error(f"❌ Document count mismatch: {len(documents)} docs != {len(products_data)} products")
                return False
            
            logger.info(f"📝 Created {len(documents)} comprehensive documents for {len(products_data)} products (1:1 ratio)")
            
            # Check if we need to clear existing data
            stats = self.index.describe_index_stats()
            current_vector_count = stats.get('total_vector_count', 0)
            
            # Only clear if we have wrong number of vectors or if forced refresh
            needs_clearing = (current_vector_count != len(products_data))
            
            if needs_clearing:
                try:
                    logger.info(f"🧹 Clearing existing Pinecone data (current: {current_vector_count}, needed: {len(products_data)})...")
                    self.index.delete(delete_all=True)
                    time.sleep(5)  # Wait for deletion
                    
                    # Verify deletion
                    stats = self.index.describe_index_stats()
                    remaining_vectors = stats.get('total_vector_count', 0)
                    if remaining_vectors > 0:
                        logger.warning(f"⚠️ {remaining_vectors} vectors still remain after deletion, waiting longer...")
                        time.sleep(5)
                    else:
                        logger.info("✅ All existing vectors successfully deleted")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error clearing existing data: {e}")
            else:
                # Check if data is actually current by comparing hashes
                if self.pinecone_manager.has_data_changed(products_data):
                    logger.info("🔄 Data has changed, updating existing vectors...")
                    needs_clearing = True
                else:
                    logger.info("✅ Data is current and vector count is correct, skipping indexing")
                    return True
            
            # Index documents with consistent IDs
            success = self.pinecone_manager._batch_add_documents(documents, "startup")
            
            if success:
                # Wait longer for final verification to avoid timing issues
                logger.info("⏳ Waiting for Pinecone to update stats...")
                time.sleep(10)  # Increased wait time
                
                # Retry verification up to 3 times
                for attempt in range(3):
                    final_stats = self.index.describe_index_stats()
                    final_count = final_stats.get('total_vector_count', 0)
                    
                    if final_count == len(products_data):
                        logger.info(f"✅ Perfect 1:1 indexing: {final_count} vectors for {len(products_data)} products")
                        return True
                    else:
                        logger.warning(f"⚠️ Attempt {attempt + 1}: Vector count {final_count} != {len(products_data)} products, waiting...")
                        if attempt < 2:  # Don't wait on last attempt
                            time.sleep(5)
                
                logger.error(f"❌ Indexing count mismatch after 3 attempts: {final_count} vectors != {len(products_data)} products")
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error in immediate Pinecone indexing: {e}")
            traceback.print_exc()
            return False
    
    def _initialize_models(self):
        """Initialize AI models"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=config.embedding_model,
                google_api_key=config.google_api_key,
                task_type="retrieval_document"
            )
            
            self.llm = ChatGoogleGenerativeAI(
                model=config.llm_model,
                temperature=config.temperature,
                google_api_key=config.google_api_key,
                max_output_tokens=config.max_output_tokens
            )
            
            logger.info("AI models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AI models: {e}")
            raise
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index"""
        try:
            self.pc = Pinecone(api_key=config.pinecone_api_key)
            
            index_name = config.pinecone_index_name
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                time.sleep(10)  # Wait for index creation
                logger.info(f"Created Pinecone index: {index_name}")
            
            self.index = self.pc.Index(index_name)
            
            # Initialize Pinecone data manager
            self.pinecone_manager = PineconeDataManager(
                self.pc, self.index, self.embeddings, self.data_handler
            )
            
            # Initialize vectorstore with correct text_key
            self.vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                text_key="page_content"
            )
            
            logger.info(f"Connected to Pinecone index: {index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise
    
    def refresh_data_before_query(self) -> bool:
        """Refresh data before each query to ensure latest information"""
        try:
            if not self.pinecone_manager:
                return False
            
            # Check if we need to refresh (time-based or force refresh every 5 minutes)
            if self.pinecone_manager.should_refresh():
                logger.info("🔄 Refreshing data before query to ensure latest information...")
                return self.pinecone_manager.refresh_pinecone_data(force_refresh=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing data before query: {e}")
            return False

    def get_answer(self, question: str, include_recommendations: bool = True) -> Dict[str, Any]:
        """Get answer with fresh data validation and optional refresh"""
        start_time = self.performance_monitor.start_timer("get_answer")
        
        try:
            # Ensure system is initialized
            if not self.is_initialized:
                logger.warning("System not initialized, attempting initialization...")
                if not self.initial_data_load():
                    return {
                        "answer": "System is still initializing. Please try again in a moment.",
                        "sources": [],
                        "response_type": "initializing",
                        "error": "System not initialized"
                    }
                self.is_initialized = True
            
            # Refresh data before query to ensure latest information
            self.refresh_data_before_query()
            
            # Check current data state
            stats = self.index.describe_index_stats() if self.index else {}
            vector_count = stats.get('total_vector_count', 0)
            
            if vector_count == 0:
                logger.warning("No vectors in Pinecone, forcing immediate refresh...")
                refresh_success = self.pinecone_manager.refresh_pinecone_data(force_refresh=True)
                if not refresh_success:
                    return {
                        "answer": "I'm still loading the product database. Please try again in a moment.",
                        "sources": [],
                        "response_type": "loading",
                        "error": "No indexed data available"
                    }
            
            # Validate inputs
            if not question or not question.strip():
                return {
                    "answer": "Please provide a valid question.",
                    "sources": [],
                    "response_type": "error",
                    "error": "Empty question"
                }
            
            # Enhanced retrieval with existing indexed data
            docs = self._enhanced_retrieval(question.strip())
            
            # Generate response
            response = self._generate_response(question.strip(), docs, include_recommendations)
            
            # Add product recommendations if enabled
            if include_recommendations and docs:
                recommendations = self._get_contextual_recommendations(docs, question)
                if recommendations:
                    response['recommendations'] = recommendations
            
            # Add metadata with data freshness info
            response.update({
                'total_products': len(self.products_data),
                'pinecone_vectors': vector_count,
                'data_ready': vector_count > 0,
                'data_freshness': self._get_data_freshness(),
                'last_refresh': self.pinecone_manager.last_refresh_time if self.pinecone_manager else None
            })
            
            # Update conversation history
            self._update_conversation_history(question, response['answer'])
            
            self.performance_monitor.end_timer("get_answer", start_time)
            return response
            
        except Exception as e:
            logger.error(f"Error in get_answer: {e}")
            traceback.print_exc()
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": [],
                "response_type": "error",
                "error": str(e)
            }
    
    def _get_data_freshness(self) -> str:
        """Get human-readable data freshness information"""
        if not self.pinecone_manager or not self.pinecone_manager.last_refresh_time:
            return "unknown"
        
        age_seconds = time.time() - self.pinecone_manager.last_refresh_time
        if age_seconds < 60:
            return f"{age_seconds:.0f} seconds ago"
        elif age_seconds < 3600:
            return f"{age_seconds/60:.0f} minutes ago"
        else:
            return f"{age_seconds/3600:.1f} hours ago"

    def _enhanced_retrieval(self, question: str) -> List[Document]:
        """Enhanced retrieval with fresh Pinecone data"""
        try:
            start_time = self.performance_monitor.start_timer("enhanced_retrieval")
            
            # Expand query for better matching
            expanded_query = self._expand_query_with_synonyms(question)
            
            # Use similarity search with score threshold
            try:
                docs = self.vectorstore.similarity_search_with_score(
                    expanded_query,
                    k=10,
                    filter={"in_stock": True}  # Only search in-stock items
                )
                
                # Filter by relevance score (keep only good matches)
                filtered_docs = []
                for doc, score in docs:
                    if score > 0.7:  # Adjust threshold as needed
                        filtered_docs.append(doc)
                
                if not filtered_docs and docs:
                    # If no high-confidence matches, take top 3 anyway
                    filtered_docs = [doc for doc, score in docs[:3]]
                
                logger.debug(f"Retrieved {len(filtered_docs)} relevant documents")
                
            except Exception as e:
                logger.warning(f"Pinecone search failed: {e}, falling back to basic search")
                # Fallback to basic search
                filtered_docs = self.vectorstore.similarity_search(expanded_query, k=5)
            
            self.performance_monitor.end_timer("enhanced_retrieval", start_time)
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error in enhanced retrieval: {e}")
            return []
    
    def _expand_query_with_synonyms(self, question: str) -> str:
        """Expand query with product-specific synonyms"""
        query_lower = question.lower()
        expanded_terms = [question]
        
        # Product type synonyms
        synonyms = {
            'camera': ['security camera', 'surveillance', 'video camera', 'webcam'],
            'headphones': ['headset', 'earphones', 'earbuds', 'audio'],
            'speaker': ['audio', 'sound system', 'bluetooth speaker'],
            'phone': ['smartphone', 'mobile', 'cell phone'],
            'laptop': ['computer', 'notebook', 'pc'],
            'watch': ['smartwatch', 'fitness tracker', 'wearable'],
            'coffee': ['espresso', 'brewing', 'caffeine'],
            'kitchen': ['cooking', 'culinary', 'chef']
        }
        
        for term, related_terms in synonyms.items():
            if term in query_lower:
                expanded_terms.extend(related_terms)
        
        return ' '.join(expanded_terms)
    
    def _generate_response(self, question: str, docs: List[Document], include_recommendations: bool = True) -> Dict[str, Any]:
        """Generate response with improved natural language and customer focus"""
        try:
            if not docs:
                # Provide helpful no-results response with suggestions
                return {
                    "answer": "I couldn't find any products matching that description. Could you try being more specific? For example, you could mention the type of device, your budget range, or any specific features you're looking for.",
                    "sources": [],
                    "response_type": "no_results",
                    "products": []
                }
            
            # Extract context and product information
            context_parts = []
            products_found = []
            
            for doc in docs:
                context_parts.append(f"Product: {doc.page_content}")
                
                # Extract product info from metadata
                if doc.metadata.get('product_id'):
                    product_info = {
                        'id': doc.metadata['product_id'],
                        'name': doc.metadata.get('product_name', ''),
                        'price': doc.metadata.get('price', 0),
                        'category': doc.metadata.get('category', ''),
                        'in_stock': doc.metadata.get('in_stock', True)
                    }
                    products_found.append(product_info)
            
            context = "\n\n".join(context_parts)
            
            # Create enhanced prompt
            prompt = self._create_enhanced_prompt(
                question, context, {}, "", "", include_recommendations
            )
            
            # Generate response using LLM
            try:
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                answer = response.content.strip()
                
                # Post-process answer to remove any technical information that slipped through
                answer = self._clean_response(answer)
                
                return {
                    "answer": answer,
                    "sources": [doc.metadata for doc in docs],
                    "response_type": "success",
                    "products": products_found,
                    "context_length": len(context)
                }
                
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                return {
                    "answer": "I found some products that might interest you, but I'm having trouble describing them right now. Please try your question again.",
                    "sources": [doc.metadata for doc in docs],
                    "response_type": "llm_error",
                    "products": products_found,
                    "error": str(e)
                }
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": "I'm having trouble processing your request. Please try again.",
                "sources": [],
                "response_type": "error",
                "error": str(e)
            }
    
    def _clean_response(self, answer: str) -> str:
        """Remove technical information and Markdown formatting from responses"""
        import re
        
        # Remove Markdown formatting
        cleaned_answer = answer
        
        # Remove bold formatting (**text** or __text__)
        cleaned_answer = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned_answer)
        cleaned_answer = re.sub(r'__(.*?)__', r'\1', cleaned_answer)
        
        # Remove italic formatting (*text* or _text_)
        cleaned_answer = re.sub(r'\*(.*?)\*', r'\1', cleaned_answer)
        cleaned_answer = re.sub(r'_(.*?)_', r'\1', cleaned_answer)
        
        # Remove bullet points and list formatting
        cleaned_answer = re.sub(r'^\s*[-*+]\s+', '', cleaned_answer, flags=re.MULTILINE)
        cleaned_answer = re.sub(r'^\s*\d+\.\s+', '', cleaned_answer, flags=re.MULTILINE)
        
        # Remove headers (# ## ###)
        cleaned_answer = re.sub(r'^#+\s+', '', cleaned_answer, flags=re.MULTILINE)
        
        # Remove code blocks and inline code
        cleaned_answer = re.sub(r'```.*?```', '', cleaned_answer, flags=re.DOTALL)
        cleaned_answer = re.sub(r'`(.*?)`', r'\1', cleaned_answer)
        
        # Remove data freshness mentions (existing functionality)
        patterns_to_remove = [
            r"Our product data is fresh as of.*?\.",
            r"The product data is fresh, last updated.*?\.",
            r"DATA FRESHNESS:.*?\.",
            r"Last updated.*?ago.*?\.",
            r"Data.*?fresh.*?\.",
            r"Database.*?updated.*?\.",
            r"Vector.*?count.*?\.",
            r"Pinecone.*?\.",
            r"System.*?initialized.*?\."
        ]
        
        for pattern in patterns_to_remove:
            cleaned_answer = re.sub(pattern, "", cleaned_answer, flags=re.IGNORECASE | re.DOTALL)
    
        # Clean up extra whitespace and line breaks
        cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()
        
        return cleaned_answer

    def _get_varied_opening_phrases(self) -> str:
        """Get varied opening phrases to avoid repetition"""
        opening_phrases = [
            "I'd be happy to help you with that!",
            "Great question!",
            "Let me help you find exactly what you're looking for!",
            "Perfect timing!",
            "I've got some excellent options for you!",
            "You're in luck!",
            "I can definitely help with that!",
            "That's a fantastic choice to explore!",
            "I'm excited to show you what we have!",
            "Let me share some great options!",
            "I have just the thing!",
            "You've come to the right place!",
            "I'd love to help you discover!",
            "Here's what I found for you!",
            "I think you'll really like this!",
            "This is going to be perfect!",
            "I'm thrilled to help!",
            "What a wonderful choice!",
            "I've got something special!",
            "Let me show you our best options!"
        ]
        
        # Filter out recently used phrases
        available_phrases = [p for p in opening_phrases if p not in self.used_opening_phrases]
        
        # If all phrases have been used recently, reset the tracking
        if not available_phrases:
            self.used_opening_phrases = []
            available_phrases = opening_phrases
        
        # Select a random phrase
        import random
        selected_phrase = random.choice(available_phrases)
        
        # Track usage
        self.used_opening_phrases.append(selected_phrase)
        if len(self.used_opening_phrases) > self.phrase_rotation_limit:
            self.used_opening_phrases.pop(0)
        
        return selected_phrase

    def _get_varied_transition_phrases(self) -> str:
        """Get varied transition phrases for when products aren't available"""
        transition_phrases = [
            "While we don't have that exact item,",
            "Although that specific product isn't available right now,",
            "I don't see that particular item in stock, but",
            "That exact model isn't available at the moment, however",
            "We're currently out of that specific item, but here's what I found:",
            "I wish I had that exact product for you, but",
            "That particular item isn't in our current inventory, though",
            "Unfortunately we don't carry that specific model, but",
            "That exact product isn't available right now, but let me show you",
            "While that particular item isn't in stock,"
        ]
        
        # Filter out recently used phrases
        available_phrases = [p for p in transition_phrases if p not in self.used_transition_phrases]
        
        # If all phrases have been used recently, reset the tracking
        if not available_phrases:
            self.used_transition_phrases = []
            available_phrases = transition_phrases
        
        # Select a random phrase
        import random
        selected_phrase = random.choice(available_phrases)
        
        # Track usage
        self.used_transition_phrases.append(selected_phrase)
        if len(self.used_transition_phrases) > self.phrase_rotation_limit:
            self.used_transition_phrases.pop(0)
        
        return selected_phrase

    def _create_enhanced_prompt(self, question: str, context: str, intent: Dict[str, Any], 
                   history: str, unavailable_info: str, 
                   include_recommendations: bool, recommendations_text: str = "") -> str:
        """Create enhanced prompt with varied conversational style and strict formatting constraints"""

        # Get varied opening phrase
        opening_phrase = self._get_varied_opening_phrases()
        
        # Get varied transition phrase for unavailable items
        transition_phrase = self._get_varied_transition_phrases()

        base_prompt = f"""You are a helpful and friendly shopping assistant for an electronics store.

CUSTOMER QUESTION: {question}

AVAILABLE PRODUCTS:
{context}

{recommendations_text}

RESPONSE GUIDELINES:
1. Start with this specific opening: "{opening_phrase}"
2. When products aren't available, use this transition: "{transition_phrase}"
3. Focus on what we DO have that meets their needs
4. Use natural, conversational language - like talking to a friend
5. Mention prices naturally and highlight savings
6. Compare products when multiple options exist
7. Keep responses conversational but informative
8. End with a helpful question or offer to assist further

STRICT FORMATTING CONSTRAINTS:
- NEVER use any markdown formatting (**bold**, *italic*, __underline__, etc.)
- NEVER use bullet points (-, *, +) or numbered lists (1., 2., 3.)
- NEVER use special characters for emphasis
- NEVER use headers (# ## ###)
- NEVER use code blocks or backticks
- Write in flowing paragraphs with natural breaks
- Use only plain text and natural punctuation
- Vary your sentence structure and avoid repetitive patterns

VOICE & TONE REQUIREMENTS:
- Sound genuinely enthusiastic and helpful
- Use varied sentence structures
- Avoid repetitive phrases or patterns
- Mix short and long sentences naturally
- Show excitement about the products
- Be conversational but professional

CONTENT RULES:
- Only mention products we actually have in stock
- Don't invent features or specifications
- If details are missing, acknowledge it helpfully
- Guide customers toward our available inventory naturally
- Make alternatives sound exciting, not like consolations
- Never mention technical system details, databases, or data freshness

RESPONSE STRUCTURE:
1. Open with the provided phrase
2. Address their specific need
3. Present our available options enthusiastically
4. Highlight key benefits and pricing
5. End with engagement (question or offer to help more)

REMEMBER: Your goal is to make customers feel excited about our available products while maintaining natural, varied conversation patterns."""
    
        return base_prompt

    def _get_contextual_recommendations(self, docs: List[Document], question: str) -> List[Dict[str, Any]]:
        """Get contextual product recommendations based on retrieved documents and question"""
        try:
            recommendations = []
            
            if not docs or not self.similarity_engine:
                return recommendations
            
            # Extract product IDs from the retrieved documents
            found_product_ids = set()
            for doc in docs:
                if doc.metadata.get('product_id'):
                    found_product_ids.add(doc.metadata['product_id'])
            
            # For each found product, get similar products
            for product_id in list(found_product_ids)[:2]:  # Limit to first 2 products
                try:
                    similar_products = self.similarity_engine.find_similar_products(
                        product_id, 
                        top_k=2,  # Get 2 similar products
                        same_category_only=False
                    )
                    
                    for similar in similar_products:
                        # Avoid recommending products already in results
                        if similar['product_id'] not in found_product_ids:
                            recommendation = {
                                'product_id': similar['product_id'],
                                'name': similar['product']['name'],
                                'price': similar['product']['price'],
                                'category': similar['product']['category'],
                                'similarity_score': similar['similarity_score'],
                                'reason': self.similarity_engine.get_recommendation_reason(
                                    self.similarity_engine.product_data[product_id],
                                    similar['product']
                                )
                            }
                            recommendations.append(recommendation)
                            
                            # Limit total recommendations
                            if len(recommendations) >= 3:
                                break
                
                except Exception as e:
                    logger.warning(f"Error getting similar products for {product_id}: {e}")
                    continue
                
                if len(recommendations) >= 3:
                    break
            
            # Sort by similarity score and return top recommendations
            recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            return recommendations[:3]
            
        except Exception as e:
            logger.error(f"Error getting contextual recommendations: {e}")
            return []

    def _update_conversation_history(self, question: str, answer: str):
        """Update conversation history for context"""
        try:
            self.conversation_history.append({
                'question': question,
                'answer': answer,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 5 conversations to avoid memory bloat
            if len(self.conversation_history) > 5:
                self.conversation_history = self.conversation_history[-5:]
                
        except Exception as e:
            logger.warning(f"Error updating conversation history: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the RAG system"""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {}
            }
            
            # Check data handler
            try:
                health_status["components"]["data_handler"] = {
                    "status": "healthy" if self.data_handler else "unhealthy",
                    "products_loaded": len(self.products_data)
                }
            except Exception as e:
                health_status["components"]["data_handler"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check Pinecone connection
            try:
                if self.index:
                    stats = self.index.describe_index_stats()
                    health_status["components"]["pinecone"] = {
                        "status": "healthy",
                        "vector_count": stats.get('total_vector_count', 0),
                        "index_name": config.pinecone_index_name
                    }
                else:
                    health_status["components"]["pinecone"] = {
                        "status": "unhealthy",
                        "error": "Index not initialized"
                    }
            except Exception as e:
                health_status["components"]["pinecone"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check LLM connection
            try:
                if self.llm:
                    # Simple test query
                    test_response = self.llm.invoke([HumanMessage(content="Test")])
                    health_status["components"]["llm"] = {
                        "status": "healthy",
                        "model": config.llm_model
                    }
                else:
                    health_status["components"]["llm"] = {
                        "status": "unhealthy",
                        "error": "LLM not initialized"
                    }
            except Exception as e:
                health_status["components"]["llm"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check embeddings
            try:
                if self.embeddings:
                    health_status["components"]["embeddings"] = {
                        "status": "healthy",
                        "model": config.embedding_model
                    }
                else:
                    health_status["components"]["embeddings"] = {
                        "status": "unhealthy",
                        "error": "Embeddings not initialized"
                    }
            except Exception as e:
                health_status["components"]["embeddings"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Overall status based on components
            unhealthy_components = [
                name for name, component in health_status["components"].items() 
                if component["status"] != "healthy"
            ]
            
            if unhealthy_components:
                health_status["status"] = "degraded" if len(unhealthy_components) < 3 else "unhealthy"
                health_status["unhealthy_components"] = unhealthy_components
            
            # Add performance metrics if available
            if hasattr(self, 'performance_monitor') and self.performance_monitor:
                health_status["performance"] = self.performance_monitor.get_stats()
            
            return health_status
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def search_products(self, query: str, filters: Dict[str, Any] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Search products with optional filters"""
        try:
            if not self.products_data:
                return []
            
            # Apply text search if query provided
            if query:
                query_lower = query.lower()
                filtered_products = []
                
                for product in self.products_data:
                    # Search in name, description, category, brand, features
                    search_fields = [
                        product.get('name', '').lower(),
                        product.get('description', '').lower(),
                        product.get('category', '').lower(),
                        product.get('brand', '').lower(),
                        ' '.join(product.get('features', [])).lower(),
                        ' '.join(product.get('tags', [])).lower()
                    ]
                    
                    if any(query_lower in field for field in search_fields):
                        filtered_products.append(product)
            else:
                filtered_products = self.products_data.copy()
            
            # Apply filters
            if filters:
                filtered_products = [p for p in filtered_products if self._apply_filters(p, filters)]
            
            # Sort by relevance (could be enhanced with embeddings)
            # For now, sort by price or name
            filtered_products.sort(key=lambda x: x.get('name', ''))
            
            return filtered_products[:limit]
            
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return []

    def _apply_filters(self, product: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filters to a product"""
        try:
            # Category filter
            if filters.get('category') and product.get('category') != filters['category']:
                return False
            
            # Brand filter
            if filters.get('brand') and product.get('brand') != filters['brand']:
                return False
            
            # Price range filters
            if filters.get('min_price') and product.get('price', 0) < filters['min_price']:
                return False
            
            if filters.get('max_price') and product.get('price', 0) > filters['max_price']:
                return False
            
            # Stock filter
            if filters.get('in_stock') is not None and product.get('inStock') != filters['in_stock']:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error applying filters: {e}")
            return True

    def get_similar_products(self, product_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get similar products using the similarity engine"""
        try:
            if not self.similarity_engine:
                return []
            
            similar = self.similarity_engine.find_similar_products(product_id, top_k=limit)
            return [s['product'] for s in similar]
            
        except Exception as e:
            logger.error(f"Error getting similar products: {e}")
            return []

def get_rag_system(data_source: Optional[Union[DataSource, str]] = None) -> Optional[EnhancedRAGSystem]:
    """Get or create the global RAG system instance - STRICT SINGLETON"""
    global _rag_system, _rag_system_lock
    
    # Prevent multiple initializations
    if _rag_system_lock:
        logger.warning("⚠️ RAG system already being initialized, returning existing instance")
        return _rag_system
    
    try:
        if _rag_system is None:
            _rag_system_lock = True
            logger.info("🚀 Initializing RAG system singleton with immediate Supabase data loading...")
            _rag_system = EnhancedRAGSystem(data_source)
            _rag_system_lock = False
        else:
            logger.debug("✅ Returning existing RAG system instance")
        
        return _rag_system
        
    except Exception as e:
        _rag_system_lock = False
        logger.error(f"❌ Error initializing RAG system: {e}")
        return None

def get_answer(question: str, data_source: Optional[Union[DataSource, str]] = None, 
               include_recommendations: bool = True) -> Dict[str, Any]:
    """Get answer with automatic data refresh and validation"""
    try:
        rag_system = get_rag_system(data_source)
        if not rag_system:
            return {
                "answer": "System not properly initialized. Please try again.",
                "sources": [],
                "response_type": "error",
                "error": "RAG system initialization failed"
            }
        
        return rag_system.get_answer(question, include_recommendations)
        
    except Exception as e:
        logger.error(f"Error in get_answer: {e}")
        return {
            "answer": "I encountered an error while processing your question. Please try again.",
            "sources": [],
            "response_type": "error",
            "error": str(e)
        }