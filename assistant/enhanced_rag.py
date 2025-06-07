import os
import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import hashlib
from collections import defaultdict
import re
import time

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
            if times:
                stats[operation] = {
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
        return stats

class EnhancedRAGSystem:
    """Enhanced RAG System with improved performance and error handling"""
    
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
        
        # Initialize embeddings and LLM
        self._initialize_models()
    
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
            
            # Load and process data
            products_data = self.data_handler.load_data(self.data_source)
            
            if not products_data:
                raise ValueError("No data loaded from source")
            
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
    
    def _create_documents(self, products_data: List[Dict[str, Any]]) -> List[Document]:
        """Create documents from product data"""
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        for product in products_data:
            try:
                # Create main product document
                content = self._create_product_content(product)
                
                # Split content if too large
                chunks = text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "product_name": product['name'],
                            "category": product['category'],
                            "price": product['price'],
                            "in_stock": product['in_stock'],
                            "doc_type": "product",
                            "chunk_id": i,
                            "product_id": product['id']
                        }
                    )
                    documents.append(doc)
                
                # Create summary document for comparisons
                if len(chunks) > 1:
                    summary_content = f"Product: {product['name']}\n"
                    summary_content += f"Summary: {product['description'][:200]}...\n"
                    summary_content += f"Price: ${product['price']}\n"
                    summary_content += f"Category: {product['category']}"
                    
                    summary_doc = Document(
                        page_content=summary_content,
                        metadata={
                            "product_name": product['name'],
                            "category": product['category'],
                            "price": product['price'],
                            "in_stock": product['in_stock'],
                            "doc_type": "product_summary",
                            "product_id": product['id']
                        }
                    )
                    documents.append(summary_doc)
                    
            except Exception as e:
                logger.warning(f"Error creating document for product {product.get('name', 'unknown')}: {e}")
                continue
        
        return documents
    
    def _create_product_content(self, product: Dict[str, Any]) -> str:
        """Create comprehensive product content"""
        content_parts = [
            f"Product: {product['name']}",
            f"Description: {product['description']}",
            f"Price: ${product['price']}",
            f"Category: {product['category']}",
            f"In Stock: {'Yes' if product['in_stock'] else 'No'}"
        ]
        
        if product.get('features'):
            content_parts.append(f"Features: {', '.join(product['features'])}")
        
        if product.get('rating'):
            content_parts.append(f"Rating: {product['rating']}/5")
        
        if product.get('reviews'):
            content_parts.append(f"Reviews: {product['reviews']} reviews")
        
        return '\n'.join(content_parts)
    
    def _generate_dynamic_keywords(self, products_data: List[Dict[str, Any]]):
        """Generate dynamic keywords from product data"""
        start_time = self.performance_monitor.start_timer("generate_keywords")
        
        self.dynamic_keywords = defaultdict(set)
        self.category_keywords = defaultdict(set)
        
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will',
            'would', 'could', 'should', 'this', 'that', 'these', 'those', 'a', 'an'
        }
        
        def extract_keywords(text: str, min_length: int = 3) -> List[str]:
            if not text:
                return []
            
            words = re.findall(r'\b\w+\b', text.lower())
            return [
                word for word in words 
                if len(word) >= min_length and word not in stop_words and not word.isdigit()
            ]
        
        for product in products_data:
            try:
                product_key = product['id']
                name_keywords = extract_keywords(product['name'])
                desc_keywords = extract_keywords(product['description'])
                
                self.dynamic_keywords[product_key].update(name_keywords)
                self.dynamic_keywords[product_key].update(desc_keywords)
                
                # Add feature keywords
                for feature in product.get('features', []):
                    feature_keywords = extract_keywords(feature)
                    self.dynamic_keywords[product_key].update(feature_keywords)
                
                # Update category keywords
                category = product['category'].lower()
                self.category_keywords[category].update(name_keywords)
                self.category_keywords[category].update(desc_keywords)
                
            except Exception as e:
                logger.warning(f"Error generating keywords for product: {e}")
                continue
        
        # Convert to lists
        self.dynamic_keywords = {k: list(v) for k, v in self.dynamic_keywords.items()}
        self.category_keywords = {k: list(v) for k, v in self.category_keywords.items()}
        
        self.performance_monitor.end_timer("generate_keywords", start_time)
        logger.info(f"Generated keywords for {len(self.dynamic_keywords)} products")
    
    def get_answer(self, question: str) -> Dict[str, Any]:
        """Get answer with comprehensive error handling and performance monitoring"""
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
            response = self._generate_response(question.strip(), docs)
            
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
    
    def _enhanced_retrieval(self, question: str, k: int = None) -> List[Document]:
        """Enhanced retrieval with performance optimization"""
        start_time = self.performance_monitor.start_timer("retrieval")
        
        try:
            if k is None:
                k = config.retrieval_k
            
            # Detect products and intent
            detected_products = self._detect_products_from_query(question)
            intent = self._detect_query_intent(question)
            
            # Adjust k based on intent
            if intent['detail_level'] == 'brief':
                k = min(k, 6)
            elif intent['wants_comparison']:
                k = min(k * 2, 20)
            
            # Perform similarity search
            docs = self.vectorstore.similarity_search(question, k=k * 2)
            
            # Filter and rank results
            filtered_docs = self._filter_and_rank_docs(docs, detected_products, intent, k)
            
            self.performance_monitor.end_timer("retrieval", start_time)
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []
    
    def _filter_and_rank_docs(self, docs: List[Document], detected_products: List[str], 
                            intent: Dict[str, Any], k: int) -> List[Document]:
        """Filter and rank documents based on relevance and availability"""
        
        available_docs = []
        unavailable_products = []
        
        # Filter by stock status
        for doc in docs:
            if doc.metadata.get('in_stock', True):
                available_docs.append(doc)
            else:
                product_name = doc.metadata.get('product_name', '')
                if product_name and product_name not in unavailable_products:
                    unavailable_products.append(product_name)
        
        # Store unavailable products for response
        if hasattr(self.vectorstore, '_unavailable_products'):
            self.vectorstore._unavailable_products = unavailable_products
        
        # Rank by relevance and diversity
        ranked_docs = self._rank_by_relevance_and_diversity(
            available_docs, detected_products, intent
        )
        
        return ranked_docs[:k]
    
    def _rank_by_relevance_and_diversity(self, docs: List[Document], 
                                       detected_products: List[str],
                                       intent: Dict[str, Any]) -> List[Document]:
        """Rank documents by relevance and ensure diversity"""
        
        scored_docs = []
        seen_products = set()
        
        for doc in docs:
            score = 0
            product_name = doc.metadata.get('product_name', '').lower()
            
            # Avoid duplicates for brief responses
            if intent['detail_level'] == 'brief' and product_name in seen_products:
                continue
            
            # Score based on product detection
            if detected_products:
                for product_id in detected_products:
                    if doc.metadata.get('product_id') == product_id:
                        score += 10
                        break
            
            # Score based on document type
            doc_type = doc.metadata.get('doc_type', '')
            if intent['wants_comparison'] and 'summary' in doc_type:
                score += 5
            elif intent['detail_level'] == 'detailed' and doc_type == 'product':
                score += 3
            
            scored_docs.append((score, doc))
            seen_products.add(product_name)
        
        # Sort by score and return documents
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs]
    
    def _detect_products_from_query(self, query: str) -> List[str]:
        """Detect products mentioned in query"""
        query_lower = query.lower()
        detected = []
        
        for product_id, keywords in self.dynamic_keywords.items():
            match_score = sum(1 for keyword in keywords if keyword in query_lower)
            if match_score >= 1:
                detected.append(product_id)
        
        return detected
    
    def _detect_query_intent(self, query: str) -> Dict[str, Any]:
        """Detect user intent from query"""
        query_lower = query.lower()
        
        intent = {
            'detail_level': 'summary',
            'wants_comparison': False,
            'wants_recommendations': False
        }
        
        # Detect detail level
        if any(word in query_lower for word in ['detailed', 'specifications', 'features', 'tell me more']):
            intent['detail_level'] = 'detailed'
        elif any(word in query_lower for word in ['quick', 'briefly', 'summary']):
            intent['detail_level'] = 'brief'
        
        # Detect comparison intent
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference', 'better']):
            intent['wants_comparison'] = True
        
        # Detect recommendation intent
        if any(word in query_lower for word in ['recommend', 'suggest', 'best', 'should i']):
            intent['wants_recommendations'] = True
        
        return intent
    
    def _generate_response(self, question: str, docs: List[Document]) -> Dict[str, Any]:
        """Generate response using LLM"""
        start_time = self.performance_monitor.start_timer("generate_response")
        
        try:
            if not docs:
                return {
                    "answer": "I don't have information about that in our current catalog.",
                    "sources": [],
                    "response_type": "no_results"
                }
            
            # Prepare context
            context = self._prepare_context(docs)
            intent = self._detect_query_intent(question)
            
            # Get unavailable products info
            unavailable_info = ""
            if hasattr(self.vectorstore, '_unavailable_products') and self.vectorstore._unavailable_products:
                unavailable_products = list(set(self.vectorstore._unavailable_products))
                if unavailable_products:
                    unavailable_info = f"Note: {', '.join(unavailable_products)} are currently out of stock."
            
            # Prepare conversation history
            history_text = self._prepare_conversation_history()
            
            # Generate prompt
            prompt = self._create_prompt(question, context, intent, history_text, unavailable_info)
            
            # Get response from LLM
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Clean response
            cleaned_response = self._clean_response(response.content)
            
            sources = [doc.metadata.get("product_name", "Unknown") for doc in docs]
            
            self.performance_monitor.end_timer("generate_response", start_time)
            
            return {
                "answer": cleaned_response,
                "sources": list(set(sources)),  # Remove duplicates
                "response_type": intent.get('detail_level', 'summary'),
                "intent": intent
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": "I encountered an error while generating the response.",
                "sources": [],
                "response_type": "error",
                "error": str(e)
            }
    
    def _prepare_context(self, docs: List[Document]) -> str:
        """Prepare context from documents"""
        context_parts = []
        seen_products = set()
        
        for i, doc in enumerate(docs):
            product_name = doc.metadata.get('product_name', '').lower()
            
            # Avoid duplicate products in context
            if product_name in seen_products:
                continue
            
            context_parts.append(f"Product {i+1}:")
            context_parts.append(doc.page_content)
            context_parts.append("")
            
            seen_products.add(product_name)
        
        return "\n".join(context_parts)
    
    def _prepare_conversation_history(self) -> str:
        """Prepare conversation history for context"""
        if not self.conversation_history:
            return ""
        
        history_parts = ["Previous conversation:"]
        for msg in self.conversation_history[-4:]:  # Last 2 exchanges
            role = "Customer" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            history_parts.append(f"{role}: {content}")
        
        return "\n".join(history_parts)
    
    def _create_prompt(self, question: str, context: str, intent: Dict[str, Any], 
                      history: str, unavailable_info: str) -> str:
        """Create optimized prompt based on intent"""
        
        base_prompt = f"""You are a friendly and knowledgeable electronics store assistant. Respond naturally and conversationally.

{history}

AVAILABLE PRODUCTS:
{context}

{unavailable_info}

CUSTOMER QUESTION: {question}

Guidelines:
- Write naturally without formatting symbols (no *, -, or markdown)
- Focus only on available products
- Be helpful and accurate
- Mention specific prices and features
"""
        
        if intent['detail_level'] == 'brief':
            base_prompt += "\nKeep response short and direct (2-3 sentences max)."
        elif intent['detail_level'] == 'detailed':
            base_prompt += "\nProvide comprehensive details about available products."
        elif intent['wants_comparison']:
            base_prompt += "\nCompare available products to help with decision making."
        
        base_prompt += "\n\nProvide a helpful response:"
        
        return base_prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean response text of formatting symbols"""
        # Remove asterisks and other formatting
        cleaned = re.sub(r'\*+', '', response)
        cleaned = re.sub(r'^[\s\-\*]+', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n[\s\-\*]+', '\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _update_conversation_history(self, question: str, answer: str):
        """Update conversation history"""
        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content=answer))
        
        # Keep only recent conversations
        if len(self.conversation_history) > config.max_conversation_history:
            self.conversation_history = self.conversation_history[-config.max_conversation_history:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "performance_metrics": self.performance_monitor.get_stats(),
            "conversation_history_length": len(self.conversation_history),
            "vector_store_initialized": self.vectorstore is not None,
            "data_source": {
                "type": self.data_source.source_type,
                "location": self.data_source.location
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health_status = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check configuration
            health_status["checks"]["config"] = config.validate()
            
            # Check data source
            try:
                test_data = self.data_handler.load_data(self.data_source)
                health_status["checks"]["data_source"] = len(test_data) > 0
            except Exception as e:
                health_status["checks"]["data_source"] = False
                health_status["errors"] = health_status.get("errors", [])
                health_status["errors"].append(f"Data source error: {str(e)}")
            
            # Check vector store
            health_status["checks"]["vector_store"] = self.vectorstore is not None
            
            # Check AI models
            try:
                test_response = self.llm.invoke([HumanMessage(content="Hello")])
                health_status["checks"]["llm"] = bool(test_response.content)
            except Exception as e:
                health_status["checks"]["llm"] = False
                health_status["errors"] = health_status.get("errors", [])
                health_status["errors"].append(f"LLM error: {str(e)}")
            
            # Overall status
            all_healthy = all(health_status["checks"].values())
            health_status["status"] = "healthy" if all_healthy else "unhealthy"
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
        
        return health_status

# Global instance
_rag_system = None

def get_rag_system(data_source: Optional[Union[DataSource, str]] = None) -> EnhancedRAGSystem:
    """Get or create RAG system instance"""
    global _rag_system
    
    if _rag_system is None or data_source is not None:
        _rag_system = EnhancedRAGSystem(data_source)
        if not _rag_system.build_vector_store():
            logger.error("Failed to build vector store")
            return None
    
    return _rag_system

def get_answer(question: str, data_source: Optional[Union[DataSource, str]] = None) -> Dict[str, Any]:
    """Main function to get answers (backward compatibility)"""
    rag_system = get_rag_system(data_source)
    if rag_system is None:
        return {
            "answer": "System initialization failed.",
            "sources": [],
            "response_type": "error",
            "error": "RAG system not available"
        }
    
    return rag_system.get_answer(question)