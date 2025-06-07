import os
import sys

def main():
    """Main entry point - launches the API server"""
    print("🚀 Starting RAG API Server...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found. Please create one with your GOOGLE_API_KEY")
        return
    
    # Import and run the Flask app
    from app import app
    app.run(host='0.0.0.0', port=5000, debug=True)

def run_tests():
    """Run RAG system tests via API"""
    print("🧪 Use the API endpoint /api/test to run tests")
    print("Example: curl http://localhost:5000/api/test")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        main()