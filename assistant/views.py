from rest_framework.decorators import api_view
from rest_framework.response import Response
from .rag import build_rag_chain
from django.shortcuts import render
import traceback

def index(request):
    return render(request, "index.html")

@api_view(['GET'])
def ask_question(request):
    query = request.GET.get('q', '')
    if not query:
        return Response({"error": "Missing 'q' parameter"}, status=400)
    
    if len(query.strip()) < 2:
        return Response({"error": "Query too short"}, status=400)

    try:
        print(f"Processing query: {query}")
        
        # Build RAG chain with better error handling
        qa_chain = build_rag_chain()
        if qa_chain is None:
            print("RAG chain is None - initialization failed")
            return Response({
                "error": "RAG system not available. Please check server logs and try again later.",
                "details": "System initialization failed"
            }, status=500)
        
        print("RAG chain obtained successfully, invoking...")
        result = qa_chain.invoke({"query": query})
        
        if not result or "result" not in result:
            print("Invalid result from RAG chain")
            return Response({
                "error": "Invalid response from AI system",
                "details": "Empty or malformed result"
            }, status=500)
        
        response_data = {
            "answer": result["result"],
            "sources": len(result.get("source_documents", [])),
            "query": query
        }
        
        print(f"Returning successful response with {response_data['sources']} sources")
        return Response(response_data)
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        print(f"Exception in ask_question: {error_msg}")
        traceback.print_exc()
        
        return Response({
            "error": error_msg,
            "details": "Server processing error",
            "query": query
        }, status=500)

@api_view(['GET'])
def system_status(request):
    """Check system health and status"""
    try:
        from .rag import validate_system_health, build_rag_chain
        
        health_status = validate_system_health()
        rag_status = build_rag_chain() is not None
        
        return Response({
            "system_health": health_status,
            "rag_available": rag_status,
            "status": "operational" if (health_status and rag_status) else "degraded"
        })
    except Exception as e:
        return Response({
            "system_health": False,
            "rag_available": False,
            "status": "error",
            "error": str(e)
        }, status=500)