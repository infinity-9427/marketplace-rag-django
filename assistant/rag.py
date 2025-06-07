import os
import json
import traceback
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from langchain_community.vectorstores.utils import filter_complex_metadata

# Load environment variables from .env file
load_dotenv()

_qa_chain = None
_vectorstore = None
_conversation_history = []

def validate_system_health():
    """Validate that all system components are working"""
    try:
        print("=== RAG System Health Check ===")
        
        # 1. Check API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ Google API key not found")
            return False
        print("✅ Google API key found")
        
        # 2. Check products.json
        current_dir = os.path.dirname(os.path.abspath(__file__))
        products_path = os.path.join(current_dir, "products.json")
        if not os.path.exists(products_path):
            print(f"❌ products.json not found at {products_path}")
            return False
        
        with open(products_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ products.json loaded successfully ({len(data)} products)")
        
        # 3. Test embeddings
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key,
                task_type="retrieval_document"
            )
            print("✅ Embeddings model accessible")
        except Exception as e:
            print(f"❌ Embeddings model failed: {e}")
            return False
        
        # 4. Test LLM
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                google_api_key=api_key,
                max_tokens=100
            )
            test_response = llm.invoke([HumanMessage(content="Hello")])
            print("✅ LLM model accessible")
        except Exception as e:
            print(f"❌ LLM model failed: {e}")
            return False
        
        print("=== All systems operational ===")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def build_rag_chain():
    global _qa_chain, _vectorstore
    if _qa_chain is not None:
        return _qa_chain
    
    # Run health check first
    if not validate_system_health():
        print("System health check failed - aborting RAG chain build")
        return None
        
    try:
        print("Starting RAG chain build process...")
        
        # Check if API key exists
        api_key = os.getenv("GOOGLE_API_KEY")
        print(f"API Key loaded: {'Yes' if api_key else 'No'}")
        if api_key:
            print(f"API Key starts with: {api_key[:10]}...")
        
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable not set")
            return None

        # Get the absolute path to products.json
        current_dir = os.path.dirname(os.path.abspath(__file__))
        products_path = os.path.join(current_dir, "products.json")
        print(f"Looking for products at: {products_path}")
        
        # Check if file exists
        if not os.path.exists(products_path):
            print(f"Error: products.json file not found at {products_path}")
            return None
        
        # Load products with error handling
        try:
            with open(products_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} products")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")
            return None
        except Exception as e:
            print(f"Error reading products file: {e}")
            return None

        # Enhanced document creation with better structure for multi-product queries
        docs = []
        print("Creating documents...")
        
        # Create a comprehensive catalog overview document
        catalog_overview = []
        catalog_overview.append("COMPLETE PRODUCT CATALOG OVERVIEW:")
        catalog_overview.append("Available Categories: Electronics, Wearables, Furniture, Smart Home, Kitchen, Gaming, Accessories")
        catalog_overview.append("\nPRODUCT SUMMARY:")
        
        for idx, item in enumerate(data):
            product_name = item.get('name', f'Product_{idx}')
            category = item.get('category', 'general')
            price = item.get('price', 0)
            rating = item.get('rating', 0)
            in_stock = item.get('inStock', True)
            
            catalog_overview.append(f"- {product_name} ({category}) - ${price} - {rating}/5 stars - {'In Stock' if in_stock else 'Out of Stock'}")
        
        # Add catalog overview document
        docs.append(Document(
            page_content="\n".join(catalog_overview),
            metadata={
                "source": "catalog_overview",
                "doc_type": "catalog_summary"
            }
        ))
        
        for idx, item in enumerate(data):
            try:
                # Debug: print the structure of the first item to understand the data format
                if idx == 0:
                    print(f"Product structure: {list(item.keys())}")
                
                # Extract comprehensive product information with safe defaults
                product_name = item.get('name', f'Product_{idx}')
                description = item.get('description', 'No description available')
                price = item.get('price', 0)
                original_price = item.get('originalPrice', None)
                category = item.get('category', 'general')
                rating = item.get('rating', 0)
                reviews = item.get('reviews', 0)
                in_stock = item.get('inStock', True)
                features = item.get('features', [])
                product_id = item.get('id', f'prod_{idx}')
                
                # Ensure features is a list and handle None values
                if not isinstance(features, list):
                    features = []
                
                # Validate required fields
                if not product_name or product_name == f'Product_{idx}':
                    print(f"Warning: Product {idx} has no valid name, skipping")
                    continue
                
                # Create main comprehensive product document
                main_content = [
                    f"PRODUCT: {product_name}",
                    f"ID: {product_id}",
                    f"Category: {category}",
                    f"Price: ${price}",
                ]
                
                # Add original price if available (indicates sale)
                if original_price and original_price != price and original_price > price:
                    savings = original_price - price
                    main_content.append(f"Original Price: ${original_price} (SALE: Save ${savings:.2f})")
                
                main_content.extend([
                    f"Rating: {rating}/5 stars ({reviews} reviews)",
                    f"Stock Status: {'Available' if in_stock else 'Out of Stock'}",
                    f"Description: {description}"
                ])
                
                # Add detailed features with better formatting
                if features:
                    main_content.append("Key Features:")
                    for feat_idx, feature in enumerate(features, 1):
                        if feature and isinstance(feature, str):
                            main_content.append(f"  {feat_idx}. {feature.strip()}")
                
                # Join all content
                full_content = "\n".join(main_content)
                
                # Validate content before creating document
                if len(full_content.strip()) < 50:  # Minimum content check
                    print(f"Warning: Product {idx} has insufficient content, skipping")
                    continue
                
                # Create main comprehensive product document
                docs.append(Document(
                    page_content=full_content,
                    metadata={
                        "source": "product_details",
                        "product_id": product_id,
                        "product_name": product_name,
                        "category": category,
                        "price": price,
                        "original_price": original_price,
                        "rating": rating,
                        "reviews": reviews,
                        "in_stock": in_stock,
                        "doc_type": "detailed_product"
                    }
                ))
                
                # Create category-specific documents for better retrieval
                category_keywords = {
                    "Electronics": ["headphones", "speaker", "audio", "sound", "wireless", "bluetooth"],
                    "Wearables": ["smartwatch", "fitness", "tracker", "watch", "wearable", "health"],
                    "Kitchen": ["coffee", "maker", "brew", "kitchen", "appliance", "thermal"],
                    "Gaming": ["keyboard", "mechanical", "gaming", "rgb", "switches", "tactile"],
                    "Smart Home": ["camera", "security", "smart", "home", "surveillance", "ai"],
                    "Furniture": ["chair", "office", "ergonomic", "desk", "furniture"],
                    "Accessories": ["charging", "wireless", "pad", "qi", "charger"]
                }
                
                keywords = category_keywords.get(category, [])
                # Convert keywords list to comma-separated string for metadata
                keywords_string = ", ".join(keywords)
                keyword_content = f"Product: {product_name} | Category: {category} | Keywords: {keywords_string} | Price: ${price} | Rating: {rating}/5 | {'In Stock' if in_stock else 'Out of Stock'} | Features: {', '.join(features[:3])}"
                
                docs.append(Document(
                    page_content=keyword_content,
                    metadata={
                        "source": "product_keywords",
                        "product_name": product_name,
                        "category": category,
                        "keywords": keywords_string,  # Changed from list to string
                        "doc_type": "keyword_focused"
                    }
                ))
                
                # Create search-optimized documents for common queries
                search_terms = []
                
                # Add product name variations
                name_parts = product_name.lower().split()
                search_terms.extend(name_parts)
                
                # Add category-specific search terms
                if category == "Electronics" and any(word in product_name.lower() for word in ["headphones", "speaker"]):
                    search_terms.extend(["music", "audio", "listening", "sound"])
                elif category == "Wearables":
                    search_terms.extend(["fitness", "health", "tracking", "smartwatch"])
                elif category == "Kitchen":
                    search_terms.extend(["coffee", "brewing", "morning", "drink"])
                elif category == "Gaming":
                    search_terms.extend(["typing", "programming", "mechanical", "keyboard"])
                elif category == "Smart Home":
                    search_terms.extend(["security", "surveillance", "home", "safety"])
                
                # Convert search_terms list to string for metadata
                search_terms_string = ", ".join(search_terms)
                search_content = f"SEARCH MATCH: {product_name} - {category} product for {search_terms_string} - ${price} - {rating}/5 stars - {description[:100]}"
                
                docs.append(Document(
                    page_content=search_content,
                    metadata={
                        "source": "search_optimized",
                        "product_name": product_name,
                        "category": category,
                        "search_terms": search_terms_string,  # Changed from list to string
                        "doc_type": "search_focused"
                    }
                ))
                
            except Exception as e:
                print(f"Error processing product {idx}: {e}")
                continue

        print(f"Created {len(docs)} total documents with enhanced variations")

        if not docs:
            print("Error: No documents were created")
            return None

        # Enhanced text splitting with better parameters for product data
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,  # Larger chunks to keep product info together
                chunk_overlap=100,  # More overlap for better context
                separators=["\n\nPRODUCT:", "\n\n", "\n", ".", " "],
                length_function=len,
                is_separator_regex=False
            )
            chunks = splitter.split_documents(docs)
            print(f"Created {len(chunks)} chunks after enhanced splitting")
        except Exception as e:
            print(f"Error during text splitting: {e}")
            return None

        # Generate unique IDs for chunks
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

        # Enhanced embeddings with better model and error handling
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key,
                task_type="retrieval_document"
            )
            print("Embeddings model initialized successfully")
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            # Fallback to a simpler embedding model
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key
                )
                print("Fallback embeddings model initialized")
            except Exception as fallback_e:
                print(f"Fallback embeddings also failed: {fallback_e}")
                return None

        # Enhanced vectorstore with better configuration
        try:
            # Filter out complex metadata before creating vector store
            filtered_chunks = []
            for chunk in chunks:
                # Create a copy of the chunk with filtered metadata
                filtered_metadata = {}
                for key, value in chunk.metadata.items():
                    # Only keep simple types that Chroma supports
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        filtered_metadata[key] = value
                    elif isinstance(value, list):
                        # Convert lists to comma-separated strings
                        if all(isinstance(item, str) for item in value):
                            filtered_metadata[key] = ", ".join(value)
                        else:
                            filtered_metadata[key] = str(value)
                    else:
                        filtered_metadata[key] = str(value)
                
                filtered_chunk = Document(
                    page_content=chunk.page_content,
                    metadata=filtered_metadata
                )
                filtered_chunks.append(filtered_chunk)
            
            _vectorstore = Chroma.from_documents(
                filtered_chunks, 
                embeddings,
                ids=chunk_ids,
                persist_directory="./chroma_db"
            )
            print("Enhanced vector store created successfully")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

        # Multi-strategy retrieval system for better multi-product queries
        def enhanced_retrieval(question: str, k: int = 15):
            """Enhanced retrieval with better handling for multi-product queries"""
            try:
                print(f"Searching for: {question}")
                
                # Detect multi-product queries
                multi_product_indicators = ['and', 'also', 'plus', 'multiple', 'several', ',', 'both', 'all']
                is_multi_product = any(indicator in question.lower() for indicator in multi_product_indicators)
                
                # Strategy 1: Get diverse results with similarity search
                initial_k = k * 3 if is_multi_product else k * 2
                similarity_docs = _vectorstore.similarity_search(
                    question, 
                    k=initial_k
                )
                
                # Strategy 2: Extract product categories/types from query
                product_keywords = {
                    'coffee': ['coffee', 'maker', 'brew', 'espresso', 'thermal'],
                    'headphones': ['headphones', 'headset', 'audio', 'wireless', 'bluetooth'],
                    'smartwatch': ['smartwatch', 'watch', 'fitness', 'tracker', 'wearable'],
                    'keyboard': ['keyboard', 'mechanical', 'gaming', 'typing'],
                    'camera': ['camera', 'security', 'surveillance', 'smart'],
                    'chair': ['chair', 'office', 'ergonomic', 'furniture'],
                    'charger': ['charger', 'charging', 'wireless', 'pad', 'qi']
                }
                
                detected_products = []
                for product_type, keywords in product_keywords.items():
                    if any(keyword in question.lower() for keyword in keywords):
                        detected_products.append(product_type)
                
                print(f"Detected product types: {detected_products}")
                
                # Strategy 3: Get category-diverse results
                categories_found = set()
                product_types_found = set()
                diverse_docs = []
                
                # First, add catalog overview for multi-product queries
                if is_multi_product or len(detected_products) > 1:
                    try:
                        catalog_docs = _vectorstore.similarity_search(
                            "catalog overview product summary", 
                            k=2,
                            filter={"doc_type": "catalog_summary"}
                        )
                        diverse_docs.extend(catalog_docs)
                    except Exception as catalog_e:
                        print(f"Warning: Could not retrieve catalog overview: {catalog_e}")
                
                # Then add diverse product documents
                for doc in similarity_docs:
                    if len(diverse_docs) >= k:
                        break
                        
                    doc_category = doc.metadata.get('category', 'unknown')
                    doc_name = doc.metadata.get('product_name', '').lower()
                    
                    # Check if this document matches detected product types
                    matches_detected = False
                    if detected_products:
                        for product_type in detected_products:
                            if any(keyword in doc_name for keyword in product_keywords.get(product_type, [])):
                                matches_detected = True
                                product_types_found.add(product_type)
                                break
                    else:
                        matches_detected = True  # Include all if no specific products detected
                    
                    # Include documents that match criteria
                    if matches_detected:
                        diverse_docs.append(doc)
                        categories_found.add(doc_category)
                
                # Strategy 4: Ensure we have representatives from each detected product type
                if detected_products and len(product_types_found) < len(detected_products):
                    missing_products = set(detected_products) - product_types_found
                    for missing_product in missing_products:
                        try:
                            specific_docs = _vectorstore.similarity_search(
                                f"{missing_product} {' '.join(product_keywords[missing_product])}", 
                                k=3
                            )
                            for doc in specific_docs:
                                if len(diverse_docs) >= k:
                                    break
                                doc_name = doc.metadata.get('product_name', '').lower()
                                if any(keyword in doc_name for keyword in product_keywords[missing_product]):
                                    diverse_docs.append(doc)
                                    break
                        except Exception as specific_e:
                            print(f"Warning: Could not retrieve specific docs for {missing_product}: {specific_e}")
                
                print(f"Found {len(diverse_docs)} documents from {len(categories_found)} categories, covering {len(product_types_found)} product types")
                return diverse_docs[:k]
                
            except Exception as e:
                print(f"Error in retrieval: {e}")
                traceback.print_exc()
                return []

        # Enhanced prompt template for multi-product queries
        enhanced_prompt = PromptTemplate(
            template="""You are an expert electronics store assistant with comprehensive knowledge of our product catalog. Your goal is to provide detailed, helpful information about products and assist customers in making informed decisions.

CONVERSATION CONTEXT:
{conversation_history}

PRODUCT CATALOG INFORMATION:
{context}

CUSTOMER QUESTION: {question}

RESPONSE GUIDELINES:

FOR SINGLE PRODUCT QUERIES:
- Provide the exact product name, full price information (including sales), detailed features, ratings, and availability
- Explain key benefits and use cases
- Compare with similar products if relevant

FOR MULTIPLE PRODUCT QUERIES:
- When customers ask for MULTIPLE products (e.g., "I need a coffee maker, headphones, and smartwatch"), provide comprehensive information about ALL requested products
- Structure your response with clear sections for each product category:
  
  **COFFEE MAKERS:**
  [Product details with name, price, features, rating, stock status]
  
  **HEADPHONES:**
  [Product details with name, price, features, rating, stock status]
  
  **SMARTWATCHES:**
  [Product details with name, price, features, rating, stock status]

ESSENTIAL INFORMATION TO INCLUDE:
- Exact product names and model information
- Current prices and any sale prices (highlight savings)
- Star ratings and number of reviews
- Key features that matter to customers
- Stock availability status
- Why each product is recommended (based on features, value, ratings)

QUALITY STANDARDS:
- Never say "I don't have information" if product details are available in the context
- Always mention sale prices and savings when applicable
- Be specific about features rather than generic
- Organize information clearly and logically
- Use natural, conversational language while being informative
- If a product is out of stock, mention it clearly and suggest alternatives if available

Provide detailed, comprehensive information about the requested products:""",
            input_variables=["context", "question", "conversation_history"]
        )

        # Enhanced LLM configuration
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,  # Lower temperature for more consistent responses
                google_api_key=api_key,
                max_tokens=1500  # More tokens for multi-product responses
            )
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            return None
        
        # Create custom retrieval chain with enhanced retrieval
        class EnhancedRetrievalQA:
            def __init__(self, llm, vectorstore, prompt_template):
                self.llm = llm
                self.vectorstore = vectorstore
                self.prompt_template = prompt_template
            
            def invoke(self, query):
                global _conversation_history
                
                try:
                    # Handle query format - could be string or dict
                    if isinstance(query, dict):
                        query_text = query.get('query', str(query))
                    else:
                        query_text = str(query)
                    
                    print(f"Processing query: {query_text}")
                    
                    # Enhanced retrieval with more documents for multi-product queries
                    docs = enhanced_retrieval(query_text, k=12)
                    print(f"Retrieved {len(docs)} documents")
                    
                    # Prepare context with better organization
                    if docs:
                        context_parts = []
                        for i, doc in enumerate(docs):
                            context_parts.append(f"=== DOCUMENT {i+1} ===")
                            context_parts.append(doc.page_content)
                            context_parts.append("")  # Empty line for separation
                        
                        context = "\n".join(context_parts)
                    else:
                        context = "No specific product information found in our catalog for this query."
                    
                    # Prepare conversation history
                    history_text = ""
                    if _conversation_history:
                        history_text = "Recent conversation:\n" + "\n".join([
                            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content[:200]}..."
                            for msg in _conversation_history[-4:]
                        ])
                    
                    # Create prompt
                    prompt_text = self.prompt_template.format(
                        context=context,
                        question=query_text,
                        conversation_history=history_text
                    )
                    
                    # Generate response
                    response = self.llm.invoke([HumanMessage(content=prompt_text)])
                    
                    # Update conversation history
                    _conversation_history.append(HumanMessage(content=query_text))
                    _conversation_history.append(AIMessage(content=response.content))
                    
                    # Keep history manageable
                    if len(_conversation_history) > 10:
                        _conversation_history = _conversation_history[-10:]
                    
                    return {
                        "result": response.content,
                        "source_documents": docs
                    }
                    
                except Exception as e:
                    print(f"Error in invoke method: {e}")
                    traceback.print_exc()
                    return {
                        "result": "I apologize, but I'm experiencing technical difficulties. Please try your question again.",
                        "source_documents": []
                    }
        
        _qa_chain = EnhancedRetrievalQA(llm, _vectorstore, enhanced_prompt)
        
        print("Enhanced RAG chain with multi-product support built successfully")
        return _qa_chain
        
    except Exception as e:
        print(f"Error building RAG chain: {e}")
        traceback.print_exc()
        return None

def get_answer(question):
    """Enhanced function to get answers with better error handling"""
    try:
        print(f"Getting answer for question: {question}")
        
        # Handle different input formats
        if isinstance(question, dict):
            question_text = question.get('query', str(question))
        else:
            question_text = str(question) if question else ""
        
        if not question_text or not question_text.strip():
            return {"error": "Empty question provided"}
        
        qa_chain = build_rag_chain()
        if qa_chain is None:
            return {"error": "RAG system not available - check logs for details"}
        
        result = qa_chain.invoke(question_text.strip())
        
        return {
            "answer": result["result"],
            "sources": len(result.get("source_documents", []))
        }
        
    except Exception as e:
        print(f"Error getting answer: {e}")
        traceback.print_exc()
        return {"error": f"Failed to process question: {str(e)}"}

def reset_conversation():
    """Reset conversation history"""
    global _conversation_history
    _conversation_history = []
    print("Conversation history reset")

# Add a test function to validate the system
def test_rag_system():
    """Test function to validate RAG system functionality"""
    try:
        print("Testing RAG system...")
        chain = build_rag_chain()
        if chain is None:
            print("RAG system failed to initialize")
            return False
        
        # Test multi-product query
        test_result = get_answer("I need a coffee maker, headphones, and smartwatch")
        if "error" in test_result:
            print(f"RAG test failed: {test_result['error']}")
            return False
        
        print("RAG system test passed successfully")
        print(f"Test response: {test_result['answer'][:200]}...")
        return True
        
    except Exception as e:
        print(f"RAG system test failed with exception: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run test when script is executed directly
    test_rag_system()