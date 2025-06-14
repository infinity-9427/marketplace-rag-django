import os
import logging
import time
from datetime import datetime
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from typing import Optional, Dict, Any
import threading
import atexit

from config import config
from data_handler import DataSource
from assistant.enhanced_rag import get_rag_system, get_answer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, 
     origins=['http://localhost:3000', 'http://localhost:3001'],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'X-Requested-With'],
     supports_credentials=True)

# Global state for warm-up
_warm_up_complete = False
_warm_up_thread = None

def warm_up_rag_system():
    """Warm up the RAG system in a background thread"""
    global _warm_up_complete
    
    try:
        logger.info("🚀 Starting RAG system warm-up...")
        rag_system = get_rag_system()
        if rag_system and rag_system.is_initialized:
            logger.info("✅ RAG system warm-up completed successfully")
            _warm_up_complete = True
        else:
            logger.error("❌ RAG system warm-up failed")
    except Exception as e:
        logger.error(f"❌ Error during RAG system warm-up: {e}")

def start_warm_up():
    """Start the warm-up process in a background thread"""
    global _warm_up_thread
    if _warm_up_thread is None or not _warm_up_thread.is_alive():
        _warm_up_thread = threading.Thread(target=warm_up_rag_system)
        _warm_up_thread.daemon = True
        _warm_up_thread.start()

# Request tracking middleware
@app.before_request
def before_request():
    g.start_time = datetime.now()
    g.request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

@app.after_request
def after_request(response):
    duration = (datetime.now() - g.start_time).total_seconds()
    logger.info(f"Request {g.request_id} completed in {duration:.3f}s - {response.status_code}")
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception in request {g.request_id}: {str(e)}")
    return jsonify({
        "error": "Internal server error",
        "status": "error",
        "request_id": g.request_id
    }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    try:
        rag_system = get_rag_system()
        if rag_system:
            health_status = rag_system.health_check()
            
            # Add system info
            health_status.update({
                "request_id": g.request_id,
                "config_valid": config.validate(),
                "system_info": {
                    "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                    "environment": os.getenv("ENVIRONMENT", "development"),
                    "warm_up_complete": _warm_up_complete
                }
            })
            
            status_code = 200 if health_status["status"] == "healthy" else 503
            return jsonify(health_status), status_code
        else:
            return jsonify({
                "status": "unhealthy",
                "error": "RAG system not initialized",
                "request_id": g.request_id
            }), 503
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "request_id": g.request_id
        }), 500

@app.route('/api/ask', methods=['POST'])
def ask_question_post():
    """Enhanced POST endpoint for asking questions with recommendations"""
    request_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000000) % 1000000}"
    start_time = time.time()
    
    try:
        # Check if warm-up is complete
        if not _warm_up_complete:
            return jsonify({
                "error": "System is still warming up. Please try again in a moment.",
                "status": "warming_up",
                "request_id": g.request_id
            }), 503
        
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
                "status": "error",
                "request_id": g.request_id
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "Invalid JSON in request body",
                "status": "error",
                "request_id": g.request_id
            }), 400
        
        # Extract question
        question = data.get('question', '').strip()
        if not question:
            return jsonify({
                "error": "Missing or empty 'question' field",
                "status": "error",
                "request_id": g.request_id
            }), 400
        
        # Extract options
        include_recommendations = data.get('include_recommendations', True)
        
        # Get answer with recommendations
        result = get_answer(question, None, include_recommendations)
        
        # Add metadata
        result.update({
            "request_id": g.request_id,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "response_time": time.time() - start_time
        })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in ask_question_post: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "status": "error",
            "request_id": g.request_id
        }), 500

@app.route('/api/ask', methods=['GET'])
def ask_question_get():
    """Enhanced GET endpoint for asking questions"""
    try:
        question = request.args.get('q', '').strip()
        if not question:
            return jsonify({
                "error": "Missing 'q' query parameter",
                "status": "error",
                "request_id": g.request_id
            }), 400
        
        # Extract options
        include_recommendations = request.args.get('recommendations', 'true').lower() == 'true'
        
        result = get_answer(question, None, include_recommendations)
        
        result.update({
            "request_id": g.request_id,
            "timestamp": datetime.now().isoformat(),
            "question": question
        })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in ask_question_get: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "status": "error",
            "request_id": g.request_id
        }), 500

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get current products data with filters"""
    try:
        rag_system = get_rag_system()
        if not rag_system:
            return jsonify({
                "error": "RAG system not initialized",
                "status": "error",
                "request_id": g.request_id
            }), 500
        
        # Extract filters from query parameters
        filters = {}
        if request.args.get('category'):
            filters['category'] = request.args.get('category')
        if request.args.get('brand'):
            filters['brand'] = request.args.get('brand')
        if request.args.get('min_price'):
            filters['min_price'] = float(request.args.get('min_price'))
        if request.args.get('max_price'):
            filters['max_price'] = float(request.args.get('max_price'))
        if request.args.get('in_stock'):
            filters['in_stock'] = request.args.get('in_stock').lower() == 'true'
        
        # Get search query
        search_query = request.args.get('search', '')
        limit = int(request.args.get('limit', 50))
        
        # Search products
        if search_query:
            products_data = rag_system.search_products(search_query, filters, limit)
        else:
            # Apply filters to all products
            products_data = rag_system.products_data
            if filters:
                products_data = [p for p in products_data if rag_system._apply_filters(p, filters)]
            products_data = products_data[:limit]
        
        return jsonify({
            "products": products_data,
            "count": len(products_data),
            "total_available": len(rag_system.products_data),
            "categories": rag_system.categories,
            "brands": rag_system.brands,
            "filters_applied": filters,
            "data_source": {
                "type": "supabase",
                "table": "products"
            },
            "status": "success",
            "request_id": g.request_id,
            "timestamp": datetime.now().isoformat()
        })
            
    except Exception as e:
        logger.error(f"Error getting products: {e}")
        return jsonify({
            "error": f"Failed to load products: {str(e)}",
            "status": "error",
            "request_id": g.request_id
        }), 500

@app.route('/api/products/<product_id>/similar', methods=['GET'])
def get_similar_products(product_id):
    """Get similar products based on embeddings"""
    try:
        rag_system = get_rag_system()
        if not rag_system:
            return jsonify({
                "error": "RAG system not initialized",
                "status": "error",
                "request_id": g.request_id
            }), 500
        
        limit = int(request.args.get('limit', 5))
        similar_products = rag_system.get_similar_products(product_id, limit)
        
        return jsonify({
            "similar_products": similar_products,
            "count": len(similar_products),
            "product_id": product_id,
            "status": "success",
            "request_id": g.request_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting similar products: {e}")
        return jsonify({
            "error": f"Failed to get similar products: {str(e)}",
            "status": "error",
            "request_id": g.request_id
        }), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all categories and brands dynamically"""
    try:
        rag_system = get_rag_system()
        if not rag_system:
            return jsonify({
                "error": "RAG system not initialized",
                "status": "error",
                "request_id": g.request_id
            }), 500
        
        return jsonify({
            "categories": rag_system.categories,
            "brands": rag_system.brands,
            "total_products": len(rag_system.products_data),
            "status": "success",
            "request_id": g.request_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return jsonify({
            "error": f"Failed to get categories: {str(e)}",
            "status": "error",
            "request_id": g.request_id
        }), 500

@app.route('/api/test', methods=['GET'])
def run_tests():
    """Enhanced test endpoint"""
    try:
        test_results = []
        test_questions = [
            "What coffee makers do you have?",
            "Show me wireless headphones",
            "What's the most expensive item?",
            "Compare gaming products",
            "Show me products under $100",
            "What kitchen appliances are available?",
            "Find me premium products"
        ]
        
        for question in test_questions:
            start_time = datetime.now()
            result = get_answer(question, include_recommendations=True)
            duration = (datetime.now() - start_time).total_seconds()
            
            test_results.append({
                "question": question,
                "success": "error" not in result,
                "response_time": f"{duration:.3f}s",
                "answer_preview": result.get('answer', result.get('error', ''))[:100] + "...",
                "products_count": len(result.get('products', [])),
                "recommendations_count": len(result.get('recommendations', [])),
                "response_type": result.get('response_type', 'unknown')
            })
        
        return jsonify({
            "message": "Tests completed",
            "results": test_results,
            "summary": {
                "total_tests": len(test_questions),
                "successful": sum(1 for r in test_results if r['success']),
                "failed": sum(1 for r in test_results if not r['success'])
            },
            "status": "success",
            "request_id": g.request_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return jsonify({
            "error": f"Test failed: {str(e)}",
            "status": "error",
            "request_id": g.request_id
        }), 500

# Initialize RAG system and start warm-up when server starts
logger.info("🚀 Starting RAG API Server...")
start_warm_up()

# Register cleanup handler
def cleanup():
    """Cleanup function to be called on shutdown"""
    logger.info("🧹 Performing cleanup on shutdown...")
    # Add any necessary cleanup code here

atexit.register(cleanup)

if __name__ == '__main__':
    if not config.validate():
        logger.error("Configuration validation failed. Check your .env file.")
        exit(1)
    
    logger.info("🚀 Starting Enhanced RAG API Server with Supabase...")
    # Use production-ready server with keep-alive
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000, threads=4, connection_limit=1000)