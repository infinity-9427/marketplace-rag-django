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
    """Enhanced RAG System with Pinecone vector store and product similarity recommendations"""
    
    def __init__(self, data_source: Optional[Union[DataSource, str]] = None):
        self.data_handler = DataHandler(enable_cache=False)  # Disable file caching
        self.performance_monitor = PerformanceMonitor()
        self.vectorstore = None
        self.qa_chain = None
        self.conversation_history = []
        self.dynamic_keywords = {}
        self.category_keywords = {}
        
        # Initialize Pinecone client
        self.pc = None
        self.index = None
        
        # Set default data source to Supabase
        if data_source is None:
            data_source = DataSource(
                source_type='supabase', 
                location='products',
                cache_duration=300
            )
        elif isinstance(data_source, str):
            if data_source.endswith('.json'):
                data_source = DataSource(source_type='file', location=data_source)
            else:
                data_source = DataSource(source_type='supabase', location=data_source)
        
        self.data_source = data_source
        
        # Initialize models and Pinecone
        self._initialize_models()
        self._initialize_pinecone()
        self.similarity_engine = ProductSimilarityEngine(self.embeddings)
    
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
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=config.pinecone_api_key)
            
            # Get or create index
            index_name = config.pinecone_index_name
            
            # Check if index exists
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
                logger.info(f"Created Pinecone index: {index_name}")
            
            # Connect to index
            self.index = self.pc.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise
    
    def update_data_source(self, new_source: Union[DataSource, str]) -> bool:
        """Update data source and rebuild vector store"""
        try:
            if isinstance(new_source, str):
                new_source = DataSource(source_type='file', location=new_source)
            
            self.data_source = new_source
            logger.info(f"Updated data source to: {new_source.location}")
            
            # Force rebuild of vector store
            return self.build_vector_store(force_rebuild=True)
            
        except Exception as e:
            logger.error(f"Error updating data source: {e}")
            return False
    
    def build_vector_store(self, force_rebuild: bool = False) -> bool:
        """Build or load vector store with Pinecone"""
        try:
            start_time = self.performance_monitor.start_timer("build_vector_store")
            logger.info("Initializing RAG system...")
            
            # Load and process data with error handling
            try:
                products_data = self.data_handler.load_data(self.data_source)
            except Exception as e:
                logger.error(f"Failed to load data from primary source: {e}")
                fallback_source = DataSource(source_type='file', location='assistant/products.json')
                try:
                    products_data = self.data_handler.load_data(fallback_source)
                    logger.info("Using fallback data source")
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    return False
            
            if not products_data:
                logger.error("No product data available")
                return False
            
            logger.info(f"Loaded {len(products_data)} products successfully")
            
            # Update similarity engine with new products
            self.similarity_engine.update_products(products_data)
            
            # Create vector store with Pinecone
            try:
                self.vectorstore = PineconeVectorStore(
                    index=self.index,
                    embedding=self.embeddings,
                    text_key="text",
                    namespace=""
                )
                logger.info("Pinecone vector store initialized")
            except Exception as e:
                logger.error(f"Error initializing Pinecone vector store: {e}")
                return False
            
            # Check if we need to rebuild
            try:
                index_stats = self.index.describe_index_stats()
                vector_count = index_stats.get('total_vector_count', 0)
                
                if force_rebuild or vector_count == 0:
                    logger.info("Populating vector store...")
                    self._populate_vector_store(products_data, force_rebuild, vector_count)
                else:
                    logger.info(f"Using existing vector store with {vector_count} vectors")
                
            except Exception as e:
                logger.error(f"Error checking/populating vector store: {e}")
                logger.warning("Continuing with limited functionality")
            
            self.performance_monitor.end_timer("build_vector_store", start_time)
            logger.info("RAG system initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            traceback.print_exc()
            return False
    
    def _populate_vector_store(self, products_data: List[Dict[str, Any]], force_rebuild: bool, vector_count: int):
        """Populate vector store with product data"""
        try:
            # Generate keywords
            self._generate_dynamic_keywords(products_data)
            
            # Create documents
            documents = self._create_documents(products_data)
            
            # Clear existing vectors if rebuilding
            if force_rebuild and vector_count > 0:
                logger.info("Clearing existing vectors...")
                self.index.delete(delete_all=True)
            
            # Add documents to Pinecone
            if documents:
                logger.info(f"Adding {len(documents)} documents to vector store...")
                self.vectorstore.add_documents(documents)
                logger.info("Documents added successfully")
        
        except Exception as e:
            logger.error(f"Error populating vector store: {e}")
            raise
    
    def _generate_dynamic_keywords(self, products_data: List[Dict[str, Any]]):
        """Generate dynamic keywords for product detection"""
        for product in products_data:
            product_id = product['id']
            keywords = []
            
            # Add name words
            name_words = product['name'].lower().split()
            keywords.extend([word for word in name_words if len(word) > 2])
            
            # Add brand
            if product.get('brand'):
                keywords.append(product['brand'].lower())
            
            # Add category
            keywords.append(product['category'].lower())
            
            # Add key features
            if product.get('features'):
                for feature in product['features'][:3]:
                    keywords.append(feature.lower())
            
            self.dynamic_keywords[product_id] = list(set(keywords))
    
    def _create_documents(self, products_data: List[Dict[str, Any]]) -> List[Document]:
        """Create documents for vector store"""
        documents = []
        
        for product in products_data:
            # Create main product document
            content = self._create_product_content(product)
            metadata = {
                'product_id': product['id'],
                'product_name': product['name'],
                'category': product['category'],
                'price': product['price'],
                'in_stock': product.get('inStock', True),
                'doc_type': 'product'
            }
            
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
            
            # Create additional documents for features and specifications
            if product.get('features'):
                feature_content = f"Product: {product['name']}\nFeatures: " + " | ".join(product['features'])
                feature_metadata = metadata.copy()
                feature_metadata['doc_type'] = 'features'
                
                documents.append(Document(
                    page_content=feature_content,
                    metadata=feature_metadata
                ))
        
        return documents
    
    def _create_product_content(self, product: Dict[str, Any]) -> str:
        """Create rich content for a product - STRICT DATA ONLY"""
        content_parts = [
            f"Product: {product['name']}",
            f"Price: ${product['price']:.2f}",
            f"Category: {product['category']}",
            f"Description: {product['description']}"
        ]
        
        if product.get('brand'):
            content_parts.append(f"Brand: {product['brand']}")
        
        if product.get('features'):
            content_parts.append(f"Features: {', '.join(product['features'])}")
        
        if not product.get('inStock', True):
            content_parts.append("Status: Out of Stock")
        
        # Add specifications if available
        if product.get('specifications'):
            specs = []
            for key, value in product['specifications'].items():
                specs.append(f"{key}: {value}")
            if specs:
                content_parts.append(f"Specifications: {', '.join(specs)}")
        
        return "\n".join(content_parts)

    def find_similar_products(self, product_name_or_id: str, top_k: int = 3) -> Dict[str, Any]:
        """Get similar products for a given product"""
        try:
            start_time = self.performance_monitor.start_timer("get_similar_products")
            
            # Find product by name or ID
            target_product = self._find_product_by_name_or_id(product_name_or_id)
            
            if not target_product:
                return {
                    "error": f"Product '{product_name_or_id}' not found",
                    "similar_products": []
                }
            
            # Get similar products
            similar_products = self.similarity_engine.find_similar_products(
                target_product['id'], top_k=top_k
            )
            
            # Format response
            recommendations = []
            for sim_data in similar_products:
                product = sim_data['product']
                reason = self.similarity_engine.get_recommendation_reason(
                    target_product, product
                )
                
                recommendations.append({
                    "name": product['name'],
                    "id": product['id'],
                    "price": product['price'],
                    "category": product['category'],
                    "similarity_score": sim_data['similarity_score'],
                    "recommendation_reason": reason,
                    "description": product['description'][:150] + "..."
                })
            
            self.performance_monitor.end_timer("get_similar_products", start_time)
            
            return {
                "target_product": {
                    "name": target_product['name'],
                    "id": target_product['id'],
                    "price": target_product['price']
                },
                "similar_products": recommendations,
                "count": len(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Error getting similar products: {e}")
            return {
                "error": str(e),
                "similar_products": []
            }
    
    def _find_product_by_name_or_id(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Find product by name or ID"""
        # First try exact ID match
        if identifier in self.similarity_engine.product_data:
            return self.similarity_engine.product_data[identifier]
        
        # Then try name matching
        identifier_lower = identifier.lower()
        for product in self.similarity_engine.product_data.values():
            if identifier_lower in product['name'].lower():
                return product
        
        return None
    
    def get_answer(self, question: str, include_recommendations: bool = True) -> Dict[str, Any]:
        """Get answer with optional product recommendations"""
        start_time = self.performance_monitor.start_timer("get_answer")
        
        try:
            # Validate inputs
            if not question or not question.strip():
                return {
                    "answer": "Please provide a valid question.",
                    "sources": [],
                    "response_type": "error",
                    "error": "Empty question"
                }
            
            # Ensure vector store is built
            if self.vectorstore is None:
                if not self.build_vector_store():
                    return {
                        "answer": "Sorry, the system is not properly initialized.",
                        "sources": [],
                        "response_type": "error",
                        "error": "Vector store initialization failed"
                    }
            
            # Enhanced retrieval
            docs = self._enhanced_retrieval(question.strip())
            
            # Generate response
            response = self._generate_response(question.strip(), docs, include_recommendations)
            
            # Add product recommendations if enabled
            if include_recommendations and docs:
                recommendations = self._get_contextual_recommendations(docs, question)
                if recommendations:
                    response['recommendations'] = recommendations
            
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
    
    def _get_contextual_recommendations(self, docs: List[Document], 
                                      question: str) -> List[Dict[str, Any]]:
        """Get contextual product recommendations based on query results"""
        try:
            # Extract products from documents and determine main product
            mentioned_products = []
            main_category = None
            primary_product_id = None
            
            # Get products from search results, prioritizing exact matches
            for doc in docs:
                product_id = doc.metadata.get('product_id')
                if product_id and product_id in self.similarity_engine.product_data:
                    product = self.similarity_engine.product_data[product_id]
                    mentioned_products.append({
                        'id': product_id,
                        'product': product,
                        'category': product['category']
                    })
                    
                    # Set primary product (first one found)
                    if primary_product_id is None:
                        primary_product_id = product_id
                        main_category = product['category']
            
            if not mentioned_products or not primary_product_id:
                logger.debug("No products found in documents for recommendations")
                return []
            
            # Get similar products using the primary product
            main_product = self.similarity_engine.product_data[primary_product_id]
            logger.debug(f"Getting recommendations for primary product: {main_product['name']}")
            
            # First try to find similar products in the same category
            similar_products = self.similarity_engine.find_similar_products(
                primary_product_id, 
                top_k=4,
                same_category_only=True
            )
            
            # If we don't have enough same-category products, expand the search
            if len(similar_products) < 2:
                broader_search = self.similarity_engine.find_similar_products(
                    primary_product_id, 
                    top_k=6,
                    same_category_only=False
                )
                # Filter by query relevance
                broader_search = self._filter_by_query_relevance(broader_search, question, main_category)
                similar_products.extend(broader_search)
            
            # Remove duplicates and filter by relevance
            seen_ids = set()
            relevant_recommendations = []
            
            for sim_data in similar_products:
                product = sim_data['product']
                product_id = product['id']
                
                # Skip duplicates and the main product itself
                if product_id in seen_ids or product_id == primary_product_id:
                    continue
                
                # Check relevance to user query
                if self._is_product_relevant_to_query(product, question, main_category):
                    reason = self.similarity_engine.get_recommendation_reason(main_product, product)
                    
                    relevant_recommendations.append({
                        "name": product['name'],
                        "id": product['id'],
                        "price": product['price'],
                        "category": product['category'],
                        "recommendation_reason": reason,
                        "similarity_score": sim_data['similarity_score']
                    })

                    seen_ids.add(product_id)
                    
                    # Limit to top 2 most relevant recommendations
                    if len(relevant_recommendations) >= 2:
                        break
            
            logger.debug(f"Found {len(relevant_recommendations)} relevant recommendations")
            return relevant_recommendations
            
        except Exception as e:
            logger.error(f"Error getting contextual recommendations: {e}")
            return []

    def _is_product_relevant_to_query(self, product: Dict[str, Any], 
                                     question: str, main_category: str) -> bool:
        """Enhanced relevance check for products"""
        query_lower = question.lower()
        
        # Extract query keywords
        query_keywords = self._extract_query_keywords(query_lower)
        
        # Calculate relevance score
        relevance = self._calculate_query_relevance(product, query_keywords, main_category)
        
        # Product is relevant if:
        # 1. Same category as main product (high relevance), OR
        # 2. High relevance score to query (0.4+ threshold), OR
        # 3. Shares important keywords with the query
        same_category = product['category'].lower() == main_category.lower()
        high_relevance = relevance > 0.4
        
        # Check for direct keyword matches in product name/description
        product_text = f"{product['name']} {product['description']}".lower()
        keyword_matches = any(keyword in product_text for keyword in query_keywords)
        
        return same_category or high_relevance or keyword_matches

    def _extract_query_keywords(self, query_lower: str) -> List[str]:
        """Extract meaningful keywords from user query"""
        # Common stop words to ignore
        stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
            'just', 'don', 'should', 'now', 'looking', 'find', 'get', 'want', 'need', 'help',
            'provide', 'show', 'tell', 'give', 'may', 'might', 'could', 'would'
        }
        
        # Extract words and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query_lower)
        keywords = [word for word in words if word not in stop_words]
        
        # Add some domain-specific keywords based on common patterns
        tech_keywords = []
        if 'noise' in query_lower and 'cancel' in query_lower:
            tech_keywords.extend(['noise-canceling', 'noise-cancelling', 'anc'])
        if 'wireless' in query_lower or 'bluetooth' in query_lower:
            tech_keywords.extend(['wireless', 'bluetooth'])
        if 'gaming' in query_lower:
            tech_keywords.append('gaming')
        if 'professional' in query_lower or 'studio' in query_lower:
            tech_keywords.extend(['professional', 'studio'])
        
        return list(set(keywords + tech_keywords))

    def _calculate_query_relevance(self, product: Dict[str, Any], 
                                 query_keywords: List[str], main_category: str) -> float:
        """Calculate how relevant a product is to the user's query"""
        relevance_score = 0.0
        
        # Create searchable text from product
        product_text = f"{product['name']} {product['description']} {product['category']}".lower()
        
        if product.get('features'):
            product_text += " " + " ".join(product['features']).lower()
        
        if product.get('brand'):
            product_text += f" {product['brand']}".lower()
        
        # Calculate keyword matches
        if query_keywords:
            matches = sum(1 for keyword in query_keywords if keyword in product_text)
            keyword_score = min(matches / len(query_keywords), 1.0)
            relevance_score += keyword_score * 0.6
        
        # Category relevance
        if product['category'].lower() == main_category.lower():
            relevance_score += 0.3
        elif any(cat_word in product['category'].lower() for cat_word in main_category.lower().split()):
            relevance_score += 0.15
        
        # Price tier relevance (products in similar price ranges are more relevant)
        # This helps avoid recommending $50 products when someone is looking at $500 products
        relevance_score += 0.1  # Base relevance for being in stock and available
        
        return min(relevance_score, 1.0)

    def _filter_by_query_relevance(self, similar_products: List[Dict[str, Any]], 
                                 question: str, main_category: str) -> List[Dict[str, Any]]:
        """Filter similar products by query relevance"""
        query_keywords = self._extract_query_keywords(question.lower())
        
        filtered_products = []
        for sim_data in similar_products:
            relevance = self._calculate_query_relevance(
                sim_data['product'], query_keywords, main_category
            )
            
            # Only include products with reasonable relevance (>0.3)
            if relevance > 0.3:
                sim_data['query_relevance'] = relevance
                filtered_products.append(sim_data)
        
        # Sort by combined score (similarity + relevance)
        filtered_products.sort(
            key=lambda x: (x['similarity_score'] * 0.7 + x['query_relevance'] * 0.3),
            reverse=True
        )
        
        return filtered_products

    def _create_enhanced_prompt(self, question: str, context: str, intent: Dict[str, Any], 
                           history: str, unavailable_info: str, 
                           include_recommendations: bool, recommendations_text: str = "") -> str:
        """Create enhanced prompt with STRICT data adherence and plain text formatting"""
        
        base_prompt = f"""You are a helpful electronics store assistant. IMPORTANT: Only use information provided in the product data below. Do not invent or hallucinate any details not explicitly stated.

{history}

AVAILABLE PRODUCTS (USE ONLY THIS DATA):
{context}

{recommendations_text}

CUSTOMER QUESTION: {question}

STRICT GUIDELINES:
- Only mention prices, features, and specifications that are explicitly listed in the product data
- DO NOT invent ratings, reviews, or customer feedback
- DO NOT add battery life, performance metrics, or technical details not in the data
- If a detail is not provided, say "I don't have that information" instead of guessing
- Focus on the actual product name, price, category, brand, and listed features only
- When recommending alternatives, explain based only on actual product differences (price, category, listed features)
- Use PLAIN TEXT ONLY - no markdown, no asterisks (*), no bullet points, no special symbols like *, -, =>, etc.
- Write in natural conversational paragraphs without any formatting symbols
- Instead of bullet points, use phrases like "First," "Second," "Also," or "Additionally"
- Do not use any special characters for emphasis or formatting

FORMATTING EXAMPLES:
WRONG: "* Product Name: $99.99 - Description here"
WRONG: "- Feature 1\n- Feature 2"
WRONG: "**Bold text**"
WRONG: "Product => Feature"

CORRECT: "The Product Name costs $99.99 and includes several features. First, it has feature 1. Additionally, it offers feature 2."
CORRECT: "I found several options for you. The first option is the Product Name at $99.99 which includes the listed features. Another good choice is the Second Product at $79.99."

Example of correct response format:
"I can help you find some great audio products for Spotify. The Smart Fitness Tracker costs $199.99 and includes GPS, heart rate monitoring, sleep analysis, and 7-day battery life. For a more budget-friendly option, there's the Basic Fitness Band at $49.99 with step tracking, calorie tracking, heart rate monitoring, and sleep tracking. Both products offer good value in their respective price ranges."
"""
    
        if include_recommendations and recommendations_text:
            base_prompt += """
- When mentioning similar products, only compare based on actual data provided
- Explain alternatives using only the features and specifications listed
- Present alternatives in flowing paragraph format without bullet points or special symbols
"""
    
        base_prompt += "\n\nProvide a helpful response using ONLY the information provided above in plain text format without any special formatting symbols:"
        
        return base_prompt

    def _enhanced_retrieval(self, question: str) -> List[Document]:
        """Enhanced retrieval with Pinecone"""
        try:
            start_time = self.performance_monitor.start_timer("enhanced_retrieval")
            
            # Expand query with synonyms for better matching
            expanded_query = self._expand_query_with_synonyms(question)
            
            # Use similarity search without score_threshold (not supported by PineconeVectorStore)
            docs = self.vectorstore.similarity_search(
                query=expanded_query,
                k=12  # Increased from 8 for better recall
            )
            
            # Extract documents and prioritize in-stock items
            in_stock_docs = []
            out_of_stock_docs = []
            
            for doc in docs:
                if doc.metadata.get('in_stock', True):
                    in_stock_docs.append(doc)
                else:
                    out_of_stock_docs.append(doc)
            
            # Prefer in-stock items but include out-of-stock as fallback
            filtered_docs = in_stock_docs[:8] if in_stock_docs else out_of_stock_docs[:8]
            
            # If still no results, try with lower k value
            if not filtered_docs:
                docs_fallback = self.vectorstore.similarity_search(
                    query=question,  # Use original question as fallback
                    k=5
                )
                filtered_docs = docs_fallback
            
            self.performance_monitor.end_timer("enhanced_retrieval", start_time)
            logger.debug(f"Retrieved {len(filtered_docs)} documents for query: {question}")
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
            'watch': ['smartwatch', 'fitness tracker', 'wearable']
        }
        
        for term, related_terms in synonyms.items():
            if term in query_lower:
                expanded_terms.extend(related_terms)
        
        return ' '.join(expanded_terms)

    def _generate_response(self, question: str, docs: List[Document], include_recommendations: bool = True) -> Dict[str, Any]:
        """Generate response from retrieved documents"""
        try:
            if not docs:
                return {
                    "answer": "I couldn't find any relevant products for your question. Please try rephrasing or ask about our available products.",
                    "sources": [],
                    "response_type": "no_results"
                }
            
            # Create context from documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Get recommendations if enabled
            recommendations_text = ""
            if include_recommendations:
                recommendations = self._get_contextual_recommendations(docs, question)
                if recommendations:
                    rec_texts = []
                    for rec in recommendations:
                        rec_texts.append(f"{rec['name']} (${rec['price']}) - {rec['recommendation_reason']}")
                    recommendations_text = f"\n\nSIMILAR/ALTERNATIVE PRODUCTS:\n" + "\n".join(rec_texts)
            
            # Create enhanced prompt
            intent = self._analyze_intent(question)
            history = self._get_conversation_context()
            prompt = self._create_enhanced_prompt(
                question, context, intent, history, "", 
                include_recommendations, recommendations_text
            )
            
            # Generate response using LLM
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "answer": answer,
                "sources": [doc.metadata for doc in docs],
                "response_type": intent.get('response_type', 'general')
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": "I encountered an error generating the response. Please try again.",
                "sources": [],
                "response_type": "error",
                "error": str(e)
            }
    
    def _analyze_intent(self, question: str) -> Dict[str, Any]:
        """Analyze user intent from question"""
        question_lower = question.lower()
        
        intent = {
            'response_type': 'general',
            'detail_level': 'brief',
            'wants_comparison': False,
            'price_focused': False
        }
        
        # Detect comparison intent
        comparison_words = ['compare', 'vs', 'versus', 'difference', 'better']
        if any(word in question_lower for word in comparison_words):
            intent['wants_comparison'] = True
            intent['response_type'] = 'comparison'
        
        # Detect detail level
        detail_words = ['detailed', 'tell me more', 'explain', 'specifications']
        if any(word in question_lower for word in detail_words):
            intent['detail_level'] = 'detailed'
        
        # Detect price focus
        price_words = ['price', 'cost', 'cheap', 'expensive', 'budget', '$']
        if any(word in question_lower for word in price_words):
            intent['price_focused'] = True
        
        return intent
    
    def _get_conversation_context(self) -> str:
        """Get recent conversation history"""
        if not self.conversation_history:
            return ""
        
        # Get last 3 exchanges
        recent_history = self.conversation_history[-3:]
        context_parts = []
        
        for entry in recent_history:
            context_parts.append(f"Previous Q: {entry['question']}")
            context_parts.append(f"Previous A: {entry['answer'][:100]}...")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _update_conversation_history(self, question: str, answer: str):
        """Update conversation history"""
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10 exchanges
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        try:
            # Check if vector store is initialized
            vectorstore_status = self.vectorstore is not None
            
            # Check Pinecone connection
            pinecone_status = False
            vector_count = 0
            try:
                if self.index:
                    stats = self.index.describe_index_stats()
                    vector_count = stats.get('total_vector_count', 0)
                    pinecone_status = True
            except Exception:
                pass
            
            # Check data source
            data_status = False
            product_count = 0
            try:
                products = self.data_handler.load_data(self.data_source)
                product_count = len(products)
                data_status = True
            except Exception:
                pass
            
            overall_status = "healthy" if all([vectorstore_status, pinecone_status, data_status]) else "unhealthy"
            
            return {
                "status": overall_status,
                "components": {
                    "vectorstore": "ok" if vectorstore_status else "error",
                    "pinecone": "ok" if pinecone_status else "error",
                    "data_source": "ok" if data_status else "error"
                },
                "metrics": {
                    "vector_count": vector_count,
                    "product_count": product_count,
                    "available_adapters": self.data_handler.get_available_adapters()
                },
                "performance": self.performance_monitor.get_stats()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.performance_monitor.get_stats()

# Global RAG system instance
_rag_system = None

def get_rag_system(data_source: Optional[Union[DataSource, str]] = None) -> Optional[EnhancedRAGSystem]:
    """Get or create the global RAG system instance"""
    global _rag_system
    
    try:
        if _rag_system is None or data_source is not None:
            logger.info("Initializing RAG system...")
            _rag_system = EnhancedRAGSystem(data_source)
            
            # Build vector store
            if not _rag_system.build_vector_store():
                logger.error("Failed to build vector store")
                return None
            
            logger.info("RAG system initialized successfully")
        
        return _rag_system
        
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        return None

def get_answer(question: str, data_source: Optional[Union[DataSource, str]] = None, 
               include_recommendations: bool = True) -> Dict[str, Any]:
    """Get answer with product recommendations"""
    try:
        rag_system = get_rag_system(data_source)
        if not rag_system:
            return {
                "answer": "System initialization failed. Please try again later.",
                "sources": [],
                "response_type": "error",
                "error": "RAG system not available"
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

def get_similar_products(product_name_or_id: str, top_k: int = 3) -> Dict[str, Any]:
    """Get similar products for a given product"""
    try:
        rag_system = get_rag_system()
        if not rag_system:
            return {
                "error": "RAG system not initialized",
                "similar_products": []
            }
        
        return rag_system.get_similar_products(product_name_or_id, top_k)
        
    except Exception as e:
        logger.error(f"Error getting similar products: {e}")
        return {
            "error": str(e),
            "similar_products": []
        }