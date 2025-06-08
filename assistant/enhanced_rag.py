import os
import json
import logging
import traceback
import re
import time
import numpy as np
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores.utils import filter_complex_metadata

from config import config
from data_handler import DataHandler, DataSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
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
            if not other_product.get('in_stock', True):
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
            score += price_similarity * 0.3
        
        # Category match
        if product1['category'] == product2['category']:
            score += 0.4
        
        # Rating similarity
        rating1 = product1.get('rating', 0)
        rating2 = product2.get('rating', 0)
        if rating1 > 0 and rating2 > 0:
            rating_similarity = 1 - abs(rating1 - rating2) / 5.0
            score += rating_similarity * 0.3
        
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
        
        # Rating comparison
        target_rating = target_product.get('rating', 0)
        similar_rating = similar_product.get('rating', 0)
        
        if similar_rating > target_rating + 0.2:
            reasons.append(f"higher rating ({similar_rating:.1f}/5)")
        
        # Category difference
        if target_product['category'] != similar_product['category']:
            reasons.append(f"alternative in {similar_product['category']}")
        
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
    """Enhanced RAG System with product similarity recommendations"""
    
    def __init__(self, data_source: Optional[Union[DataSource, str]] = None):
        self.data_handler = DataHandler()
        self.performance_monitor = PerformanceMonitor()
        self.vectorstore = None
        self.qa_chain = None
        self.conversation_history = []
        self.dynamic_keywords = {}
        self.category_keywords = {}
        
        # Set default data source
        if data_source is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(current_dir, "products.json")
            data_source = DataSource(source_type='file', location=default_path)
        elif isinstance(data_source, str):
            data_source = DataSource(source_type='file', location=data_source)
        
        self.data_source = data_source
        
        # Initialize models and similarity engine
        self._initialize_models()
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
        """Build or load vector store with caching"""
        try:
            start_time = self.performance_monitor.start_timer("build_vector_store")
            
            # Generate cache key based on data source
            cache_key = hashlib.md5(
                f"{self.data_source.location}_{self.data_source.source_type}".encode()
            ).hexdigest()
            vector_store_path = f"{config.vector_db_path}_{cache_key}"
            
            # Load and process data
            products_data = self.data_handler.load_data(self.data_source)
            
            if not products_data:
                raise ValueError("No data loaded from source")
            
            # Update similarity engine with new products
            self.similarity_engine.update_products(products_data)
            
            # Check if vector store exists and is recent
            if not force_rebuild and os.path.exists(vector_store_path):
                try:
                    self.vectorstore = Chroma(
                        persist_directory=vector_store_path,
                        embedding_function=self.embeddings
                    )
                    logger.info(f"Loaded existing vector store from {vector_store_path}")
                    self.performance_monitor.end_timer("build_vector_store", start_time)
                    return True
                except Exception as e:
                    logger.warning(f"Error loading existing vector store: {e}")
            
            # Generate keywords
            self._generate_dynamic_keywords(products_data)
            
            # Create documents
            documents = self._create_documents(products_data)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=vector_store_path
            )
            self.vectorstore.persist()
            
            logger.info(f"Created vector store with {len(documents)} documents")
            self.performance_monitor.end_timer("build_vector_store", start_time)
            return True
            
        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            traceback.print_exc()
            return False
    
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
                for feature in product['features'][:3]:  # Top 3 features
                    feature_words = feature.lower().split()
                    keywords.extend([word for word in feature_words if len(word) > 3])
            
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
                'in_stock': product.get('in_stock', True),
                'doc_type': 'product',
                'rating': product.get('rating', 0)
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
        """Create rich content for a product"""
        content_parts = [
            f"Product: {product['name']}",
            f"Price: ${product['price']:.2f}",
            f"Category: {product['category']}",
            f"Description: {product['description']}"
        ]
        
        if product.get('brand'):
            content_parts.append(f"Brand: {product['brand']}")
        
        if product.get('rating'):
            content_parts.append(f"Rating: {product['rating']}/5")
        
        if product.get('features'):
            content_parts.append(f"Features: {', '.join(product['features'])}")
        
        if not product.get('in_stock', True):
            content_parts.append("Status: Out of Stock")
        
        return "\n".join(content_parts)
    
    def get_similar_products(self, product_name_or_id: str, top_k: int = 3) -> Dict[str, Any]:
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
                    "rating": product.get('rating', 0),
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
        """Create enhanced prompt with better recommendation guidance"""
        
        base_prompt = f"""You are a friendly and knowledgeable electronics store assistant. Respond naturally and conversationally.

{history}

AVAILABLE PRODUCTS:
{context}

{recommendations_text}

{unavailable_info}

CUSTOMER QUESTION: {question}

Guidelines:
- Write naturally without formatting symbols (no *, -, or markdown)
- Focus on products that directly answer the customer's question
- Be helpful and accurate with specific prices and features
- When mentioning products from "SIMILAR/ALTERNATIVE PRODUCTS", explain WHY they're good alternatives
- Always stay within the same product category (headphones with headphones, cameras with cameras, etc.)
- Use natural language like "You might also consider the [product name] which offers [specific benefit]"
- Compare key features that matter to the customer (price, battery life, quality, etc.)
- If a customer asks about noise-canceling headphones, suggest other noise-canceling headphones
- Don't suggest unrelated products

Example good recommendation:
"The Wireless Noise-Canceling Headphones at $299.99 have excellent 30-hour battery and 4.8/5 rating. You might also consider the Studio Pro Headphones at $249.99 which offer similar noise cancellation but with a more compact design."
"""
        
        if include_recommendations and recommendations_text:
            base_prompt += """
- Always mention relevant alternatives that enhance the customer's options
- Explain the key differences and benefits clearly
- Help customers understand their choices within the product category they're interested in
"""
        
        if intent.get('detail_level') == 'brief':
            base_prompt += "\nKeep response concise but informative."
        elif intent.get('detail_level') == 'detailed':
            base_prompt += "\nProvide comprehensive details and thorough comparisons."
        elif intent.get('wants_comparison'):
            base_prompt += "\nFocus on detailed comparisons between related products."
        
        base_prompt += "\n\nProvide a helpful response with relevant product recommendations:"
        
        return base_prompt

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

# Add missing methods to EnhancedRAGSystem class
def _enhanced_retrieval(self, question: str) -> List[Document]:
    """Enhanced retrieval with multiple strategies"""
    try:
        start_time = self.performance_monitor.start_timer("enhanced_retrieval")
        
        # Use similarity search with metadata filtering
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 8,
                "filter": {"in_stock": True}  # Only get in-stock products
            }
        )
        
        docs = retriever.get_relevant_documents(question)
        
        # If no results, try without stock filter
        if not docs:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}
            )
            docs = retriever.get_relevant_documents(question)
        
        self.performance_monitor.end_timer("enhanced_retrieval", start_time)
        return docs
        
    except Exception as e:
        logger.error(f"Error in enhanced retrieval: {e}")
        return []

def _generate_response(self, question: str, docs: List[Document], 
                      include_recommendations: bool = True) -> Dict[str, Any]:
    """Generate response with contextual recommendations"""
    try:
        start_time = self.performance_monitor.start_timer("generate_response")
        
        # Create context from documents
        context = self._create_context_from_docs(docs)
        
        # Get intent and conversation history
        intent = self._analyze_intent(question)
        history = self._get_conversation_context()
        
        # Get recommendations if enabled
        recommendations_text = ""
        if include_recommendations and docs:
            recommendations = self._get_contextual_recommendations(docs, question)
            if recommendations:
                recommendations_text = self._format_recommendations_for_prompt(recommendations)
        
        # Create enhanced prompt
        prompt = self._create_enhanced_prompt(
            question, context, intent, history, "", 
            include_recommendations, recommendations_text
        )
        
        # Generate response
        response = self.llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()
        
        # Extract sources
        sources = [
            {
                "product_name": doc.metadata.get('product_name', 'Unknown'),
                "category": doc.metadata.get('category', 'Unknown'),
                "price": doc.metadata.get('price', 0),
                "content": doc.page_content[:200] + "..."
            }
            for doc in docs[:3]
        ]
        
        self.performance_monitor.end_timer("generate_response", start_time)
        
        return {
            "answer": answer,
            "sources": sources,
            "response_type": "success",
            "intent": intent
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            "answer": "I apologize, but I encountered an error while generating a response.",
            "sources": [],
            "response_type": "error",
            "error": str(e)
        }

def _create_context_from_docs(self, docs: List[Document]) -> str:
    """Create context string from documents"""
    if not docs:
        return "No relevant products found."
    
    context_parts = []
    seen_products = set()
    
    for doc in docs[:6]:  # Limit to top 6 docs
        product_name = doc.metadata.get('product_name', 'Unknown Product')
        
        # Avoid duplicates
        if product_name in seen_products:
            continue
        seen_products.add(product_name)
        
        # Format product info
        context_parts.append(f"Product: {product_name}")
        context_parts.append(f"Price: ${doc.metadata.get('price', 0):.2f}")
        context_parts.append(f"Category: {doc.metadata.get('category', 'Unknown')}")
        context_parts.append(f"Details: {doc.page_content[:300]}")
        context_parts.append("---")
    
    return "\n".join(context_parts)

def _format_recommendations_for_prompt(self, recommendations: List[Dict[str, Any]]) -> str:
    """Format recommendations for inclusion in prompt"""
    if not recommendations:
        return ""
    
    rec_parts = ["SIMILAR/ALTERNATIVE PRODUCTS:"]
    
    for rec in recommendations:
        rec_parts.append(
            f"- {rec['name']} (${rec['price']:.2f}) - {rec['recommendation_reason']}"
        )
    
    return "\n".join(rec_parts) + "\n"

def _analyze_intent(self, question: str) -> Dict[str, Any]:
    """Analyze user intent from question"""
    question_lower = question.lower()
    
    intent = {
        'wants_comparison': any(word in question_lower for word in ['compare', 'vs', 'versus', 'difference', 'better']),
        'wants_recommendation': any(word in question_lower for word in ['recommend', 'suggest', 'best', 'good']),
        'price_sensitive': any(word in question_lower for word in ['cheap', 'affordable', 'budget', 'price', 'cost']),
        'detail_level': 'detailed' if any(word in question_lower for word in ['detail', 'spec', 'feature']) else 'brief'
    }
    
    return intent

def _get_conversation_context(self) -> str:
    """Get recent conversation context"""
    if not self.conversation_history:
        return ""
    
    # Get last 2 exchanges
    recent_history = self.conversation_history[-4:]  # 2 Q&A pairs
    
    context_parts = ["RECENT CONVERSATION:"]
    for i in range(0, len(recent_history), 2):
        if i + 1 < len(recent_history):
            context_parts.append(f"Q: {recent_history[i]}")
            context_parts.append(f"A: {recent_history[i+1][:150]}...")
    
    return "\n".join(context_parts) + "\n" if len(context_parts) > 1 else ""

def _update_conversation_history(self, question: str, answer: str):
    """Update conversation history"""
    self.conversation_history.extend([question, answer])
    
    # Keep only last 10 exchanges (20 items)
    if len(self.conversation_history) > 20:
        self.conversation_history = self.conversation_history[-20:]

def health_check(self) -> Dict[str, Any]:
    """System health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check vector store
        if self.vectorstore:
            try:
                # Test query
                test_docs = self.vectorstore.similarity_search("test", k=1)
                health_status["components"]["vector_store"] = {
                    "status": "healthy",
                    "document_count": len(test_docs)
                }
            except Exception as e:
                health_status["components"]["vector_store"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        else:
            health_status["components"]["vector_store"] = {
                "status": "not_initialized"
            }
            health_status["status"] = "unhealthy"
        
        # Check similarity engine
        if self.similarity_engine and self.similarity_engine.product_data:
            health_status["components"]["similarity_engine"] = {
                "status": "healthy",
                "product_count": len(self.similarity_engine.product_data)
            }
        else:
            health_status["components"]["similarity_engine"] = {
                "status": "not_initialized"
            }
            if health_status["status"] == "healthy":
                health_status["status"] = "degraded"
        
        # Check AI models
        try:
            test_embedding = self.embeddings.embed_query("test")
            health_status["components"]["embeddings"] = {
                "status": "healthy",
                "embedding_dimension": len(test_embedding)
            }
        except Exception as e:
            health_status["components"]["embeddings"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "unhealthy"
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def get_performance_stats(self) -> Dict[str, Any]:
    """Get performance statistics"""
    return {
        "performance_stats": self.performance_monitor.get_stats(),
        "timestamp": datetime.now().isoformat()
    }

# Add missing methods to EnhancedRAGSystem class
EnhancedRAGSystem._enhanced_retrieval = _enhanced_retrieval
EnhancedRAGSystem._generate_response = _generate_response
EnhancedRAGSystem._create_context_from_docs = _create_context_from_docs
EnhancedRAGSystem._format_recommendations_for_prompt = _format_recommendations_for_prompt
EnhancedRAGSystem._analyze_intent = _analyze_intent
EnhancedRAGSystem._get_conversation_context = _get_conversation_context
EnhancedRAGSystem._update_conversation_history = _update_conversation_history
EnhancedRAGSystem.health_check = health_check
EnhancedRAGSystem.get_performance_stats = get_performance_stats