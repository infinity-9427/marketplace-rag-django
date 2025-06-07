import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from assistant.rag import get_answer, build_rag_chain, test_super_accurate_rag

app = Flask(__name__)

# Configure CORS with specific settings
CORS(app, 
     origins=['http://localhost:3000'],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])

# Global variable to store the RAG chain
rag_chain = None

def initialize_rag():
    """Initialize the RAG system once at startup"""
    global rag_chain
    if rag_chain is None:
        print("🤖 Initializing RAG System...")
        rag_chain = build_rag_chain()
        if rag_chain:
            print("✅ RAG System Ready!")
        else:
            print("❌ Failed to initialize RAG system")
    return rag_chain is not None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "RAG API is running",
        "initialized": rag_chain is not None
    })

@app.route('/api/ask', methods=['POST'])
def ask_question_post():
    """POST endpoint for asking questions"""
    try:
        # Check if RAG system is initialized
        if not initialize_rag():
            return jsonify({
                "error": "RAG system not initialized. Check your .env file and Google API key.",
                "status": "error"
            }), 500
        
        # Get question from request body
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                "error": "Missing 'question' in request body",
                "status": "error"
            }), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({
                "error": "Question cannot be empty",
                "status": "error"
            }), 400
        
        # Get answer from RAG system
        result = get_answer(question)
        
        if "error" in result:
            return jsonify({
                "error": result['error'],
                "status": "error"
            }), 500
        
        # Return successful response
        return jsonify({
            "question": question,
            "answer": result['answer'],
            "sources": result['sources'],
            "response_type": result['response_type'],
            "intent": result.get('intent', {}),
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/ask', methods=['GET'])
def ask_question_get():
    """GET endpoint for asking questions (alternative to POST)"""
    try:
        # Check if RAG system is initialized
        if not initialize_rag():
            return jsonify({
                "error": "RAG system not initialized. Check your .env file and Google API key.",
                "status": "error"
            }), 500
        
        # Get question from query parameter
        question = request.args.get('q', '').strip()
        if not question:
            return jsonify({
                "error": "Missing 'q' query parameter",
                "status": "error"
            }), 400
        
        # Get answer from RAG system
        result = get_answer(question)
        
        if "error" in result:
            return jsonify({
                "error": result['error'],
                "status": "error"
            }), 500
        
        # Return successful response
        return jsonify({
            "question": question,
            "answer": result['answer'],
            "sources": result['sources'],
            "response_type": result['response_type'],
            "intent": result.get('intent', {}),
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/test', methods=['GET'])
def run_tests():
    """Test endpoint to verify RAG system functionality"""
    try:
        if not initialize_rag():
            return jsonify({
                "error": "RAG system not initialized",
                "status": "error"
            }), 500
        
        # Run tests
        test_results = []
        test_questions = [
            "What headphones do you have?",
            "Tell me about wireless products",
            "What's the most expensive item?",
            "Compare gaming products"
        ]
        
        for question in test_questions:
            result = get_answer(question)
            test_results.append({
                "question": question,
                "success": "error" not in result,
                "answer": result.get('answer', result.get('error', ''))[:100] + "...",
                "sources": result.get('sources', [])
            })
        
        return jsonify({
            "message": "Tests completed",
            "results": test_results,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Test failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get all available products (optional endpoint)"""
    try:
        import json
        current_dir = os.path.dirname(os.path.abspath(__file__))
        products_path = os.path.join(current_dir, "assistant", "products.json")
        
        with open(products_path, 'r', encoding='utf-8') as f:
            products_data = json.load(f)
        
        return jsonify({
            "products": products_data,
            "count": len(products_data),
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to load products: {str(e)}",
            "status": "error"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "status": "error"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "status": "error"
    }), 500

if __name__ == '__main__':
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found. Please create one with your GOOGLE_API_KEY")
    else:
        print("🚀 Starting RAG API Server...")
        app.run(host='0.0.0.0', port=5000, debug=True)