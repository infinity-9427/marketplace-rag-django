import os
import json
import traceback
from collections import defaultdict
import re
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_community.vectorstores.utils import filter_complex_metadata

# Load environment variables
load_dotenv()

# Global variables
_qa_chain = None
_vectorstore = None
_conversation_history = []
_dynamic_keywords = {}
_category_keywords = {}

def generate_dynamic_keywords(products_data):
    """Generate dynamic keywords based on actual product data"""
    global _dynamic_keywords, _category_keywords
    
    _dynamic_keywords = defaultdict(set)
    _category_keywords = defaultdict(set)
    
    def extract_keywords_from_text(text, min_length=3):
        """Extract meaningful keywords from text"""
        if not text:
            return []
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will',
            'would', 'could', 'should', 'this', 'that', 'these', 'those', 'a', 'an'
        }
        
        keywords = []
        for word in words:
            if len(word) >= min_length and word not in stop_words and not word.isdigit():
                keywords.append(word)
        
        return keywords
    
    for product in products_data:
        try:
            name = product.get('name', '').lower()
            description = product.get('description', '').lower()
            category = product.get('category', 'general').lower()
            features = product.get('features', [])
            
            product_key = name.replace(' ', '_').replace('-', '_')
            
            name_keywords = extract_keywords_from_text(name)
            _dynamic_keywords[product_key].update(name_keywords)
            
            desc_keywords = extract_keywords_from_text(description)
            _dynamic_keywords[product_key].update(desc_keywords)
            
            for feature in features:
                if isinstance(feature, str):
                    feature_keywords = extract_keywords_from_text(feature)
                    _dynamic_keywords[product_key].update(feature_keywords)
            
            _category_keywords[category].update(name_keywords)
            _category_keywords[category].update(desc_keywords)
            
            category_synonyms = {
                'electronics': ['electronic', 'device', 'gadget', 'tech', 'digital'],
                'wearables': ['wearable', 'smart', 'fitness', 'health', 'monitor'],
                'furniture': ['furniture', 'office', 'desk', 'ergonomic', 'comfortable'],
                'kitchen': ['kitchen', 'cooking', 'appliance', 'food', 'beverage'],
                'gaming': ['gaming', 'game', 'player', 'competitive', 'performance'],
                'smart home': ['smart', 'home', 'automation', 'control', 'security'],
                'accessories': ['accessory', 'add-on', 'complement', 'enhancement']
            }
            
            if category in category_synonyms:
                _dynamic_keywords[product_key].update(category_synonyms[category])
                _category_keywords[category].update(category_synonyms[category])
            
        except Exception:
            continue
    
    _dynamic_keywords = {k: list(v) for k, v in _dynamic_keywords.items()}
    _category_keywords = {k: list(v) for k, v in _category_keywords.items()}
    
    return _dynamic_keywords, _category_keywords

def detect_products_from_query(query):
    """Dynamically detect products mentioned in query using generated keywords"""
    global _dynamic_keywords, _category_keywords
    
    query_lower = query.lower()
    detected_products = []
    detected_categories = []
    
    for product_key, keywords in _dynamic_keywords.items():
        match_score = 0
        for keyword in keywords:
            if keyword in query_lower:
                match_score += 1
        
        if match_score >= 1:
            detected_products.append({
                'product_key': product_key,
                'keywords': keywords,
                'match_score': match_score
            })
    
    for category, keywords in _category_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                detected_categories.append(category)
                break
    
    detected_products.sort(key=lambda x: x['match_score'], reverse=True)
    
    return detected_products, detected_categories

def enhanced_retrieval(question: str, k: int = 15):
    """Enhanced retrieval with dynamic product detection"""
    try:
        multi_product_indicators = ['and', 'also', 'plus', 'multiple', 'several', ',', 'both', 'all']
        is_multi_product = any(indicator in question.lower() for indicator in multi_product_indicators)
        
        detected_products, detected_categories = detect_products_from_query(question)
        
        initial_k = k * 3 if is_multi_product else k * 2
        similarity_docs = _vectorstore.similarity_search(question, k=initial_k)
        
        categories_found = set()
        product_types_found = set()
        diverse_docs = []
        
        if is_multi_product or len(detected_products) > 1:
            try:
                catalog_docs = _vectorstore.similarity_search(
                    "catalog overview product summary", 
                    k=2,
                    filter={"doc_type": "catalog_summary"}
                )
                diverse_docs.extend(catalog_docs)
            except Exception:
                pass
        
        for doc in similarity_docs:
            if len(diverse_docs) >= k:
                break
                
            doc_category = doc.metadata.get('category', 'unknown').lower()
            doc_name = doc.metadata.get('product_name', '').lower()
            
            matches_detected = False
            if detected_products:
                for detected in detected_products:
                    keywords = detected['keywords']
                    if any(keyword in doc_name or keyword in doc.page_content.lower() for keyword in keywords):
                        matches_detected = True
                        product_types_found.add(detected['product_key'])
                        break
            else:
                matches_detected = True
            
            if matches_detected:
                diverse_docs.append(doc)
                categories_found.add(doc_category)
        
        if detected_products and len(product_types_found) < len(detected_products):
            missing_products = [p for p in detected_products if p['product_key'] not in product_types_found]
            for missing_product in missing_products[:3]:
                try:
                    search_terms = ' '.join(missing_product['keywords'][:5])
                    specific_docs = _vectorstore.similarity_search(search_terms, k=3)
                    
                    for doc in specific_docs:
                        if len(diverse_docs) >= k:
                            break
                        doc_name = doc.metadata.get('product_name', '').lower()
                        if any(keyword in doc_name for keyword in missing_product['keywords']):
                            diverse_docs.append(doc)
                            break
                except Exception:
                    pass
        
        return diverse_docs[:k]
        
    except Exception:
        return []

def validate_system_health():
    """Validate that all system components are working"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return False
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        products_path = os.path.join(current_dir, "products.json")
        if not os.path.exists(products_path):
            return False
        
        with open(products_path, 'r', encoding='utf-8') as f:
            json.load(f)
        
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key,
                task_type="retrieval_document"
            )
        except Exception:
            return False
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                google_api_key=api_key,
                max_output_tokens=100
            )
            llm.invoke([HumanMessage(content="Hello")])
        except Exception:
            return False
        
        return True
        
    except Exception:
        return False

def detect_query_intent(query):
    """Detect user intent for response detail level"""
    query_lower = query.lower()
    
    detailed_keywords = [
        'detailed', 'details', 'comprehensive', 'in-depth', 'explain', 'how does',
        'specifications', 'specs', 'features', 'tell me more', 'elaborate',
        'thorough', 'complete information', 'full description', 'everything about'
    ]
    
    comparison_keywords = [
        'compare', 'comparison', 'vs', 'versus', 'difference', 'better', 'best',
        'which one', 'choose between', 'recommend', 'similar'
    ]
    
    summary_keywords = [
        'quick', 'briefly', 'summary', 'overview', 'just tell me', 'short',
        'main features', 'key points', 'simply', 'concise'
    ]
    
    intent = {
        'detail_level': 'summary',
        'wants_comparison': False,
        'wants_recommendations': False
    }
    
    if any(keyword in query_lower for keyword in detailed_keywords):
        intent['detail_level'] = 'detailed'
    elif any(keyword in query_lower for keyword in summary_keywords):
        intent['detail_level'] = 'brief'
    
    if any(keyword in query_lower for keyword in comparison_keywords):
        intent['wants_comparison'] = True
    
    if any(word in query_lower for word in ['recommend', 'suggest', 'best', 'should i']):
        intent['wants_recommendations'] = True
    
    return intent

def create_enhanced_prompt_templates():
    """Create human-like prompt templates without formatting symbols"""
    
    base_context = """You are a friendly and knowledgeable electronics store assistant. Respond naturally as if you're having a conversation with a customer in the store. Be helpful, accurate, and conversational.

CONVERSATION HISTORY:
{conversation_history}

PRODUCT INFORMATION:
{context}

CUSTOMER QUESTION: {question}

RESPONSE STYLE GUIDELINES:
- Write naturally like you're talking to a friend
- Never use asterisks, bullet points, or any formatting symbols
- Never use markdown formatting like ** or * or - for lists
- Keep responses concise and easy to read
- Mention specific prices, features, and availability
- Be enthusiastic but not pushy
- Use natural transitions between products
- Write in flowing paragraphs without special formatting"""

    templates = {
        'brief': PromptTemplate(
            template=base_context + """

Keep your response very short and direct. Just give the essential information the customer needs in plain text. Maximum 2-3 sentences total with no formatting symbols.

Give a quick, helpful answer:""",
            input_variables=["context", "question", "conversation_history"]
        ),
        
        'summary': PromptTemplate(
            template=base_context + """

Provide a friendly, conversational response with key product details. For each product, mention the name, current price, top features that matter most to customers, and whether it's in stock. Keep it natural and flowing like you're explaining to someone in person. Use plain text only, no formatting symbols. Aim for 3-5 sentences per product.

Give a helpful, conversational summary:""",
            input_variables=["context", "question", "conversation_history"]
        ),
        
        'detailed': PromptTemplate(
            template=base_context + """

Since the customer wants detailed information, provide comprehensive details in a natural, conversational way. Include all specifications, features, benefits, use cases, and technical details. Write as if you're a knowledgeable friend sharing everything they should know about the product. Use only plain text with no special formatting.

Provide detailed, comprehensive information:""",
            input_variables=["context", "question", "conversation_history"]
        ),
        
        'comparison': PromptTemplate(
            template=base_context + """

Help the customer compare products by naturally explaining the key differences, strengths of each option, and which might work better for different needs. Write conversationally as if you're helping them weigh their options in person. Use plain text only.

Provide a helpful comparison to guide their decision:""",
            input_variables=["context", "question", "conversation_history"]
        )
    }
    
    return templates

class SuperAccurateRetrievalQA:
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        self.prompt_templates = create_enhanced_prompt_templates()
    
    def invoke(self, query):
        global _conversation_history
        
        try:
            if isinstance(query, dict):
                query_text = query.get('query', str(query))
            else:
                query_text = str(query)
            
            intent = detect_query_intent(query_text)
            docs = self.enhanced_retrieval_with_intent(query_text, intent, k=12)
            
            if docs:
                context_parts = []
                seen_products = set()
                
                for i, doc in enumerate(docs):
                    product_name = doc.metadata.get('product_name', '').lower()
                    if product_name and product_name in seen_products and intent['detail_level'] == 'brief':
                        continue
                    
                    if product_name:
                        seen_products.add(product_name)
                    
                    context_parts.append(f"Product {i+1}:")
                    context_parts.append(doc.page_content)
                    context_parts.append("")
                
                context = "\n".join(context_parts)
            else:
                context = "I don't have specific information about that product in our current catalog."
            
            history_text = ""
            if _conversation_history:
                history_text = "Previous conversation:\n" + "\n".join([
                    f"{'Customer' if isinstance(msg, HumanMessage) else 'Me'}: {msg.content[:100]}..."
                    for msg in _conversation_history[-2:]
                ])
            
            template_key = 'summary'
            if intent['wants_comparison']:
                template_key = 'comparison'
            elif intent['detail_level'] == 'brief':
                template_key = 'brief'
            elif intent['detail_level'] == 'detailed':
                template_key = 'detailed'
            
            prompt_template = self.prompt_templates[template_key]
            
            prompt_text = prompt_template.format(
                context=context,
                question=query_text,
                conversation_history=history_text
            )
            
            temperature = 0.3 if intent['detail_level'] == 'brief' else 0.4
            max_output_tokens = {
                'brief': 200,
                'summary': 600,
                'detailed': 1000,
                'comparison': 800
            }.get(template_key, 600)
            
            query_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                max_output_tokens=max_output_tokens
            )
            
            response = query_llm.invoke([HumanMessage(content=prompt_text)])
            
            # Clean response to remove any asterisks or formatting symbols
            cleaned_response = response.content
            cleaned_response = re.sub(r'\*+', '', cleaned_response)  # Remove asterisks
            cleaned_response = re.sub(r'^[\s\-\*]+', '', cleaned_response, flags=re.MULTILINE)  # Remove bullet points
            cleaned_response = re.sub(r'\n[\s\-\*]+', '\n', cleaned_response)  # Clean line starts
            
            _conversation_history.append(HumanMessage(content=query_text))
            _conversation_history.append(AIMessage(content=cleaned_response))
            
            if len(_conversation_history) > 6:
                _conversation_history = _conversation_history[-6:]
            
            return {
                "result": cleaned_response,
                "source_documents": docs,
                "intent": intent,
                "template_used": template_key
            }
            
        except Exception:
            return {
                "result": "Sorry, I'm having some technical difficulties right now. Could you try asking again?",
                "source_documents": [],
                "intent": {},
                "template_used": "error"
            }
    
    def enhanced_retrieval_with_intent(self, question: str, intent: dict, k: int = 15):
        """Enhanced retrieval that considers user intent"""
        try:
            if intent['detail_level'] == 'brief':
                k = min(k, 6)
            elif intent['detail_level'] == 'detailed':
                k = min(k * 2, 18)
            
            docs = enhanced_retrieval(question, k)
            
            if intent['wants_comparison']:
                comparison_docs = []
                single_product_docs = []
                
                for doc in docs:
                    if 'catalog_summary' in doc.metadata.get('doc_type', ''):
                        comparison_docs.insert(0, doc)
                    elif 'detailed_product' in doc.metadata.get('doc_type', ''):
                        comparison_docs.append(doc)
                    else:
                        single_product_docs.append(doc)
                
                docs = comparison_docs + single_product_docs[:max(0, k - len(comparison_docs))]
            
            return docs[:k]
            
        except Exception:
            return enhanced_retrieval(question, k)

def build_rag_chain():
    """Build and return the RAG chain"""
    global _qa_chain, _vectorstore
    
    try:
        # Validate system health first
        if not validate_system_health():
            print("❌ System health check failed")
            return None
        
        # Load products and generate keywords
        current_dir = os.path.dirname(os.path.abspath(__file__))
        products_path = os.path.join(current_dir, "products.json")
        
        with open(products_path, 'r', encoding='utf-8') as f:
            products_data = json.load(f)
        
        generate_dynamic_keywords(products_data)
        
        # Initialize embeddings and LLM
        api_key = os.getenv("GOOGLE_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
            task_type="retrieval_document"
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=api_key,
            max_output_tokens=1000
        )
        
        # Create or load vector store
        chroma_dir = os.path.join(os.path.dirname(current_dir), "chroma_db")
        
        if os.path.exists(chroma_dir):
            _vectorstore = Chroma(
                persist_directory=chroma_dir,
                embedding_function=embeddings
            )
        else:
            # Create vector store from products
            documents = []
            for product in products_data:
                content = f"Product: {product['name']}\n"
                content += f"Description: {product['description']}\n"
                content += f"Price: ${product['price']}\n"
                content += f"Category: {product['category']}\n"
                
                if product.get('features'):
                    content += f"Features: {', '.join(product['features'])}\n"
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "product_name": product['name'],
                        "category": product['category'],
                        "price": product['price'],
                        "doc_type": "product"
                    }
                )
                documents.append(doc)
            
            # Create vector store
            _vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=chroma_dir
            )
            _vectorstore.persist()
        
        # Create RAG chain
        _qa_chain = SuperAccurateRetrievalQA(llm, _vectorstore)
        
        return _qa_chain
        
    except Exception as e:
        print(f"❌ Error building RAG chain: {e}")
        traceback.print_exc()
        return None

def get_answer(question: str):
    """Main function to get answers from the RAG system"""
    try:
        global _qa_chain
        if _qa_chain is None:
            _qa_chain = build_rag_chain()
            if _qa_chain is None:
                return {
                    "answer": "Sorry, the system is not properly configured. Please check your setup.",
                    "sources": [],
                    "response_type": "error",
                    "error": "System initialization failed"
                }
        
        result = _qa_chain.invoke(question)
        
        return {
            "answer": result.get("result", "No answer available"),
            "sources": [doc.metadata.get("product_name", "Unknown") for doc in result.get("source_documents", [])],
            "response_type": result.get("template_used", "unknown"),
            "intent": result.get("intent", {})
        }
        
    except Exception as e:
        return {
            "answer": "Sorry, I encountered an error processing your question.",
            "sources": [],
            "response_type": "error",
            "error": str(e)
        }

def test_super_accurate_rag():
    """Test the RAG system with sample queries"""
    test_questions = [
        "What headphones do you have?",
        "Tell me about wireless products",
        "What's the most expensive item?",
        "Compare gaming products"
    ]
    
    print("Testing RAG System...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        result = get_answer(question)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Answer: {result['answer'][:100]}...")
            print(f"📚 Sources: {result['sources']}")
    
    print("\n✅ Testing completed!")