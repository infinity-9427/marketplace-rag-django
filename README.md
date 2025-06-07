# RAG Product Catalog System

A Retrieval-Augmented Generation (RAG) system for intelligent product catalog queries using Google's Gemini AI and vector search capabilities.

## 🚀 Tech Stack

- **Backend**: Python Flask API
- **AI/ML**: Google Gemini AI (LLM), LangChain (RAG framework)
- **Vector Store**: ChromaDB for semantic search
- **Data Sources**: JSON files, REST APIs with caching
- **Embeddings**: Google Text Embedding Model

## 📋 Features

- **Intelligent Query Processing**: Natural language product search and recommendations
- **Multi-Response Types**: Brief, detailed, and comparison responses based on user intent
- **Dynamic Keyword Extraction**: Automatically extracts product features and categories
- **Conversation History**: Maintains context across queries
- **Performance Monitoring**: Built-in metrics and health checks
- **Flexible Data Sources**: Support for JSON files and REST APIs with caching
- **Stock-Aware Responses**: Filters out-of-stock items and suggests alternatives

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Google API Key (for Gemini AI)

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

3. **Configure environment**:
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
```

4. **Start the server**:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## 📡 API Reference

### Health Check
```bash
curl http://localhost:5000/health
```

### Ask Questions (POST)
```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What wireless headphones do you have under $200?"
  }'
```

### Ask Questions (GET)
```bash
curl "http://localhost:5000/api/ask?q=Show me gaming products"
```

### Get Products
```bash
curl http://localhost:5000/api/products
```

### Performance Stats
```bash
curl http://localhost:5000/api/performance
```

### Run Tests
```bash
curl http://localhost:5000/api/test
```

### Update Data Source
```bash
curl -X POST http://localhost:5000/api/data-source \
  -H "Content-Type: application/json" \
  -d '{
    "type": "api",
    "location": "https://api.example.com/products",
    "headers": {"Authorization": "Bearer token"},
    "cache_duration": 3600
  }'
```

## 🎯 RAG System Purpose

This RAG system serves as an intelligent product catalog assistant that:

1. **Understands Natural Language**: Processes customer queries in plain English
2. **Provides Contextual Responses**: Considers conversation history and user intent
3. **Maintains Data Accuracy**: Uses real-time product information with stock availability
4. **Optimizes User Experience**: Delivers brief, detailed, or comparative responses based on query type
5. **Handles Complex Queries**: Supports product comparisons, feature searches, and price-based filtering

### Query Examples

```bash
# Product search
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Do you have noise-canceling headphones?"}'

# Price-based filtering
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me products under $100"}'

# Product comparison
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare gaming keyboards vs regular keyboards"}'

# Feature-specific search
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What products have wireless charging?"}'
```

## 🔧 Development Commands

### Run with Debug Mode
```bash
python app.py
```

### Test the System
```bash
# Via API endpoint
curl http://localhost:5000/api/test

# Or direct test script
python -c "
import requests
response = requests.get('http://localhost:5000/api/test')
print(response.json())
"
```

### Monitor Performance
```bash
# Get performance metrics
curl http://localhost:5000/api/performance | jq '.'

# Health check with details
curl http://localhost:5000/health | jq '.'
```

### Clear Cache
```bash
rm -rf cache/ chroma_db*/
```

## 📁 Project Structure

```
rag/
├── app.py                    # Flask API server
├── config.py                # Configuration management
├── data_handler.py          # Multi-source data loading
├── assistant/
│   ├── enhanced_rag.py      # Core RAG implementation
│   └── products.json        # Sample product data
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create this)
└── README.md               # This file
```

## 🚨 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini AI API key | Yes |
| `ENVIRONMENT` | Environment name (development/production) | No |

## 📊 Response Types

The system automatically detects user intent and responds accordingly:

- **Brief**: Quick, 2-3 sentence responses for simple queries
- **Detailed**: Comprehensive product information with all specifications
- **Comparison**: Side-by-side product comparisons
- **Summary**: Balanced overview of available products

## 🔍 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GOOGLE_API_KEY` is set in `.env`
2. **Port Conflicts**: Change port in `app.py` if 5000 is in use
3. **Cache Issues**: Clear cache with `rm -rf cache/ chroma_db*/`

### Debug Mode
```bash
# Enable detailed logging
export FLASK_DEBUG=1
python app.py
```