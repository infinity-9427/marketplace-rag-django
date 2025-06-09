# RAG Product Catalog System

A sophisticated Retrieval-Augmented Generation (RAG) system for intelligent product catalog queries using Google's Gemini AI, Pinecone vector database, and advanced semantic search capabilities.

## 🧠 What is RAG (Retrieval-Augmented Generation)?

RAG is an AI architecture that combines the power of large language models (LLMs) with external knowledge retrieval. This system works in three key phases:

1. **🔍 Retrieval**: Search through a vector database to find relevant product information
2. **🔗 Augmentation**: Combine retrieved data with the user's query to create context
3. **✨ Generation**: Use an LLM to generate natural, conversational responses

### Why RAG for Product Catalogs?

- **Real-time Data**: Always uses current product information from your database
- **Semantic Understanding**: Understands intent behind queries like "affordable wireless headphones"
- **Contextual Responses**: Provides personalized recommendations based on user preferences
- **Accurate Information**: Prevents AI hallucination by grounding responses in actual data

## 🚀 Tech Stack

- **Backend**: Python Flask API with CORS support
- **AI/ML**: Google Gemini AI (LLM- gemini-2.0-flash), LangChain (RAG framework)
- **Vector Store**: Pinecone for semantic search and embeddings
- **Data Sources**: Supabase (PostgreSQL), JSON files, REST APIs with intelligent caching
- **Embeddings**: Google Text Embedding Model (text-embedding-004)
- **Similarity Engine**: Custom product recommendation system with cosine similarity

## 🎯 RAG System Architecture

```
User Query → Query Processing → Vector Search → Context Retrieval → LLM Generation → Response
     ↓              ↓               ↓              ↓               ↓            ↓
"Show me        Extract         Find similar    Retrieve       Generate      "I found 3
headphones"     keywords        products in     product        natural       great wireless
                               vector space     details        response      headphones..."
```

### Core Components

1. **[`EnhancedRAGSystem`](assistant/enhanced_rag.py)**: Main orchestrator managing the entire RAG pipeline
2. **[`PineconeDataManager`](assistant/enhanced_rag.py)**: Handles vector database synchronization with 1:1 product mapping
3. **[`ProductSimilarityEngine`](assistant/enhanced_rag.py)**: Advanced product recommendation system
4. **[`DataHandler`](data_handler.py)**: Multi-source data loading with intelligent caching
5. **[`DatabaseManager`](dbConnection.py)**: Supabase integration with health monitoring

## 📋 Features

### 🤖 Intelligent Query Processing
- **Natural Language Understanding**: Processes queries like "wireless headphones under $200"
- **Intent Detection**: Automatically determines if user wants brief info, detailed specs, or comparisons
- **Semantic Search**: Finds products based on meaning, not just keyword matching
- **Query Expansion**: Enhances searches with synonyms and related terms

### 🎭 Multi-Response Types
- **Brief**: Quick, 2-3 sentence responses for simple queries
- **Detailed**: Comprehensive product information with specifications
- **Comparison**: Side-by-side analysis of multiple products
- **Summary**: Balanced overview of available product categories

### 🔄 Dynamic Keyword Extraction
- **Feature Analysis**: Automatically extracts product features and benefits
- **Category Intelligence**: Understands product hierarchies and relationships
- **Price Range Categorization**: Groups products into logical price segments
- **Stock Awareness**: Real-time inventory consideration in recommendations

### 💬 Conversation History
- **Context Maintenance**: Remembers previous queries in the conversation
- **Follow-up Support**: Handles questions like "show me cheaper alternatives"
- **Preference Learning**: Adapts recommendations based on user behavior

### 📊 Performance Monitoring
- **Response Time Tracking**: Monitors query processing performance
- **Health Checks**: Comprehensive system status monitoring
- **Vector Database Sync**: Ensures data consistency between sources
- **Cache Management**: Intelligent caching to optimize response times

### 🔌 Flexible Data Sources
- **Supabase Integration**: Real-time PostgreSQL database connection
- **JSON File Support**: Static product catalogs for development/testing
- **REST API Integration**: Dynamic data loading from external APIs
- **Automatic Caching**: Configurable cache duration for performance optimization

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Google API Key (for Gemini AI)
- Pinecone API Key (for vector database)
- Supabase Account (for product data)

### Setup

1. **Clone and navigate to project**:
```bash
git clone <repository-url>
cd rag
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**:
```bash
# Create .env file from template
cp .env.example .env

# Edit .env with your API keys
GOOGLE_API_KEY=your_google_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-ai-voice
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

4. **Initialize Pinecone Index** (First time setup):
```python
# Run this once to create your Pinecone index
python -c "
from assistant.enhanced_rag import get_rag_system
rag = get_rag_system()
print('Pinecone index initialized successfully!')
"
```

5. **Start the server**:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## 📡 API Reference

### Health Check
```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "pinecone_status": "connected",
  "supabase_status": "connected",
  "product_count": 150,
  "vector_count": 150,
  "data_freshness": "2 minutes ago"
}
```

### Ask Questions (POST)
```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What wireless headphones do you have under $200?",
    "include_recommendations": true
  }'
```

**Response:**
```json
{
  "answer": "I found some great wireless headphones under $200 for you! The Sony WH-CH720N offers excellent noise cancellation at $149.99...",
  "products": [...],
  "recommendations": [...],
  "response_type": "detailed",
  "data_freshness": "Real-time",
  "processing_time": "1.2s"
}
```

### Ask Questions (GET)
```bash
curl "http://localhost:5000/api/ask?q=Show me gaming products&recommendations=true"
```

### Get Products with Filters
```bash
curl "http://localhost:5000/api/products?category=Electronics&min_price=50&max_price=300&in_stock=true&limit=20"
```

### Get Similar Products
```bash
curl http://localhost:5000/api/products/product_123/similar?limit=5
```

### Performance Statistics
```bash
curl http://localhost:5000/api/performance
```

### Run System Tests
```bash
curl http://localhost:5000/api/test
```

## 🎯 RAG System Deep Dive

### How the RAG Pipeline Works

1. **Query Processing** ([`get_answer`](assistant/enhanced_rag.py)):
   - Receives user question
   - Triggers data refresh if needed
   - Preprocesses and expands query

2. **Vector Retrieval** ([`_enhanced_retrieval`](assistant/enhanced_rag.py)):
   - Converts query to embeddings
   - Searches Pinecone vector database
   - Retrieves semantically similar products

3. **Context Assembly** ([`_generate_response`](assistant/enhanced_rag.py)):
   - Combines retrieved products with query
   - Applies conversation history
   - Generates recommendation context

4. **Response Generation** ([`_create_enhanced_prompt`](assistant/enhanced_rag.py)):
   - Creates optimized prompt for Gemini AI
   - Ensures natural, conversational tone
   - Formats response without technical details

### Vector Database Strategy

The system uses a **1:1 product-to-document mapping** in Pinecone:

- Each product becomes exactly one comprehensive vector document
- Includes all product metadata (name, description, features, price, etc.)
- Enables precise semantic matching and similarity calculations
- Prevents data duplication and maintains consistency

### Recommendation Engine

The [`ProductSimilarityEngine`](assistant/enhanced_rag.py) provides:

- **Embedding-based Similarity**: Cosine similarity between product vectors
- **Contextual Scoring**: Price range, category, and brand considerations
- **Intelligent Reasoning**: Explains why products are recommended
- **Diverse Suggestions**: Balances similar and complementary products

### Query Examples

```bash
# Natural language product search
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Do you have noise-canceling headphones?"}'

# Price-based filtering with context
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me the best products under $100"}'

# Product comparison requests
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare gaming keyboards vs regular keyboards"}'

# Feature-specific searches
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What products have wireless charging capability?"}'

# Follow-up conversations
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me something cheaper"}'
```

## 🔧 Development Commands

### Run with Debug Mode
```bash
export FLASK_DEBUG=1
python app.py
```

### Test the RAG System
```bash
# Via API endpoint
curl http://localhost:5000/api/test

# Direct test with multiple queries
python -c "
import requests
import json

test_queries = [
    'What coffee makers do you have?',
    'Show me wireless headphones under $150',
    'Compare gaming vs office chairs',
    'What is your most expensive product?'
]

for query in test_queries:
    response = requests.post('http://localhost:5000/api/ask',
                           json={'question': query})
    result = response.json()
    print(f'Q: {query}')
    print(f'A: {result.get(\"answer\", \"Error\")[:100]}...')
    print('---')
"
```

### Monitor Performance
```bash
# Get detailed performance metrics
curl http://localhost:5000/api/performance | jq '.'

# Health check with full system status
curl http://localhost:5000/health | jq '.'

# Monitor real-time logs
tail -f app.log
```

### Clear Cache and Reset
```bash
# Clear local cache
rm -rf cache/ __pycache__/

# Reset Pinecone index (if needed)
python -c "
from assistant.enhanced_rag import get_rag_system
rag = get_rag_system()
rag.pinecone_manager.refresh_pinecone_data(force_refresh=True)
"
```

## 📁 Project Structure

```
rag/
├── app.py                    # Flask API server with CORS and error handling
├── config.py                # Configuration management and validation
├── data_handler.py          # Multi-source data loading with caching
├── dbConnection.py          # Supabase database integration
├── assistant/
│   ├── enhanced_rag.py      # Core RAG implementation with Pinecone
│   └── products.json        # Sample product data for testing
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create from .env.example)
├── .env.example            # Environment template
├── .gitignore              # Git ignore patterns
└── README.md               # This comprehensive guide
```

## 🚨 Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GOOGLE_API_KEY` | Google Gemini AI API key | Yes | - |
| `PINECONE_API_KEY` | Pinecone vector database API key | Yes | - |
| `PINECONE_INDEX_NAME` | Pinecone index name | Yes | `rag-ai-voice` |
| `SUPABASE_URL` | Supabase project URL | Yes | - |
| `SUPABASE_KEY` | Supabase anonymous key | Yes | - |
| `ENVIRONMENT` | Environment name | No | `development` |

## 📊 Response Types & AI Behavior

The system automatically detects user intent and responds accordingly:

### Brief Responses
- **Trigger**: Simple, direct questions
- **Format**: 2-3 sentences with key product highlights
- **Example**: "What headphones do you have?" → Quick overview of available options

### Detailed Responses
- **Trigger**: Specific product inquiries or technical questions
- **Format**: Comprehensive product information with specifications
- **Example**: "Tell me about the Sony WH-1000XM4" → Full feature breakdown

### Comparison Responses
- **Trigger**: "Compare", "vs", "difference between"
- **Format**: Side-by-side analysis with pros/cons
- **Example**: "Gaming vs office chairs" → Detailed comparison table

### Summary Responses
- **Trigger**: Category or broad searches
- **Format**: Balanced overview of product range
- **Example**: "Kitchen appliances" → Overview of categories and price ranges

## 🔍 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   # Verify your API keys are set
   python -c "
   import os
   from dotenv import load_dotenv
   load_dotenv()
   print('Google API Key:', 'SET' if os.getenv('GOOGLE_API_KEY') else 'MISSING')
   print('Pinecone API Key:', 'SET' if os.getenv('PINECONE_API_KEY') else 'MISSING')
   print('Supabase URL:', 'SET' if os.getenv('SUPABASE_URL') else 'MISSING')
   "
   ```

2. **Port Conflicts**
   ```bash
   # Change port in app.py if 5000 is in use
   sed -i 's/port=5000/port=5001/' app.py
   ```

3. **Pinecone Index Issues**
   ```bash
   # Recreate Pinecone index
   python -c "
   from assistant.enhanced_rag import get_rag_system
   rag = get_rag_system()
   rag._initialize_pinecone()
   print('Pinecone index recreated')
   "
   ```

4. **Database Connection Issues**
   ```bash
   # Test Supabase connection
   python -c "
   from dbConnection import get_database_manager
   db = get_database_manager()
   health = db.health_check()
   print('Database Status:', health['status'])
   "
   ```

### Debug Mode
```bash
# Enable detailed logging
export FLASK_DEBUG=1
export LOG_LEVEL=DEBUG
python app.py
```

### Performance Optimization

1. **Enable Caching**: Set appropriate cache durations in [`data_handler.py`](data_handler.py)
2. **Optimize Queries**: Use specific product filters to reduce vector search scope
3. **Monitor Memory**: Large product catalogs may require memory optimization
4. **Batch Processing**: For bulk operations, use the batch endpoints

## 🚀 Advanced Usage

### Custom Data Sources

You can extend the system to work with your own data sources:

```python
from data_handler import DataSource
from assistant.enhanced_rag import get_rag_system

# Custom API data source
custom_source = DataSource(
    source_type='api',
    location='https://your-api.com/products',
    headers={'Authorization': 'Bearer your-token'},
    cache_duration=3600
)

rag = get_rag_system(custom_source)
```

### Extending the RAG Pipeline

The system is designed to be extensible. Key extension points:

1. **Custom Embeddings**: Modify [`_initialize_models`](assistant/enhanced_rag.py) in `EnhancedRAGSystem`
2. **Query Processing**: Extend [`_enhanced_retrieval`](assistant/enhanced_rag.py) for custom search logic
3. **Response Generation**: Customize [`_create_enhanced_prompt`](assistant/enhanced_rag.py) for different response styles
4. **Recommendation Logic**: Enhance [`ProductSimilarityEngine`](assistant/enhanced_rag.py) for domain-specific recommendations

## 📈 Monitoring & Analytics

The system provides comprehensive monitoring through:

- **Performance Metrics**: Query response times, vector search performance
- **Health Monitoring**: Database connectivity, API status, data freshness
- **Usage Analytics**: Query patterns, popular products, recommendation effectiveness
- **Error Tracking**: Detailed logging for debugging and optimization

Access monitoring data via:
```bash
curl http://localhost:5000/api/performance
curl http://localhost:5000/health
```

---

**Built with ❤️ using Python, LangChain, Google Gemini AI, and Pinecone**

