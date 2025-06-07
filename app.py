import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from typing import Optional, Dict, Any

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
                    "environment": os.getenv("ENVIRONMENT", "development")
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
    """Enhanced POST endpoint for asking questions"""
    try:
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
        
        # Extract optional data source
        data_source = None
        if 'data_source' in data:
            ds_config = data['data_source']
            if isinstance(ds_config, str):
                data_source = ds_config
            elif isinstance(ds_config, dict):
                data_source = DataSource(
                    source_type=ds_config.get('type', 'file'),
                    location=ds_config.get('location', ''),
                    headers=ds_config.get('headers'),
                    auth=ds_config.get('auth'),
                    cache_duration=ds_config.get('cache_duration', 3600)
                )
        
        # Get answer
        result = get_answer(question, data_source)
        
        # Add metadata
        result.update({
            "request_id": g.request_id,
            "timestamp": datetime.now().isoformat(),
            "question": question
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
        
        # Optional data source from query params
        data_source = None
        ds_type = request.args.get('source_type')
        ds_location = request.args.get('source_location')
        
        if ds_type and ds_location:
            data_source = DataSource(
                source_type=ds_type,
                location=ds_location
            )
        
        result = get_answer(question, data_source)
        
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

@app.route('/api/data-source', methods=['POST'])
def update_data_source():
    """Update data source endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "Invalid JSON in request body",
                "status": "error",
                "request_id": g.request_id
            }), 400
        
        # Create data source
        if isinstance(data, str):
            data_source = DataSource(source_type='file', location=data)
        else:
            data_source = DataSource(
                source_type=data.get('type', 'file'),
                location=data.get('location', ''),
                headers=data.get('headers'),
                auth=data.get('auth'),
                cache_duration=data.get('cache_duration', 3600)
            )
        
        # Update RAG system
        rag_system = get_rag_system()
        if rag_system:
            success = rag_system.update_data_source(data_source)
            
            if success:
                return jsonify({
                    "message": "Data source updated successfully",
                    "status": "success",
                    "data_source": {
                        "type": data_source.source_type,
                        "location": data_source.location
                    },
                    "request_id": g.request_id
                })
            else:
                return jsonify({
                    "error": "Failed to update data source",
                    "status": "error",
                    "request_id": g.request_id
                }), 500
        else:
            return jsonify({
                "error": "RAG system not initialized",
                "status": "error",
                "request_id": g.request_id
            }), 500
            
    except Exception as e:
        logger.error(f"Error updating data source: {e}")
        return jsonify({
            "error": f"Failed to update data source: {str(e)}",
            "status": "error",
            "request_id": g.request_id
        }), 500

@app.route('/api/performance', methods=['GET'])
def get_performance_stats():
    """Get performance statistics"""
    try:
        rag_system = get_rag_system()
        if rag_system:
            stats = rag_system.get_performance_stats()
            stats.update({
                "request_id": g.request_id,
                "timestamp": datetime.now().isoformat()
            })
            return jsonify(stats)
        else:
            return jsonify({
                "error": "RAG system not initialized",
                "status": "error",
                "request_id": g.request_id
            }), 500
            
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        return jsonify({
            "error": str(e),
            "status": "error",
            "request_id": g.request_id
        }), 500

@app.route('/api/test', methods=['GET'])
def run_tests():
    """Enhanced test endpoint"""
    try:
        test_results = []
        test_questions = [
            "What headphones do you have?",
            "Tell me about wireless products",
            "What's the most expensive item?",
            "Compare gaming products",
            "Show me products under $100"
        ]
        
        for question in test_questions:
            start_time = datetime.now()
            result = get_answer(question)
            duration = (datetime.now() - start_time).total_seconds()
            
            test_results.append({
                "question": question,
                "success": "error" not in result,
                "response_time": f"{duration:.3f}s",
                "answer_preview": result.get('answer', result.get('error', ''))[:100] + "...",
                "sources_count": len(result.get('sources', [])),
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

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get current products data"""
    try:
        rag_system = get_rag_system()
        if rag_system:
            # Load current products
            products_data = rag_system.data_handler.load_data(rag_system.data_source)
            
            return jsonify({
                "products": products_data,
                "count": len(products_data),
                "data_source": {
                    "type": rag_system.data_source.source_type,
                    "location": rag_system.data_source.location
                },
                "status": "success",
                "request_id": g.request_id,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "error": "RAG system not initialized",
                "status": "error",
                "request_id": g.request_id
            }), 500
            
    except Exception as e:
        logger.error(f"Error getting products: {e}")
        return jsonify({
            "error": f"Failed to load products: {str(e)}",
            "status": "error",
            "request_id": g.request_id
        }), 500

if __name__ == '__main__':
    if not config.validate():
        logger.error("Configuration validation failed. Check your .env file.")
        exit(1)
    
    logger.info("🚀 Starting Enhanced RAG API Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)