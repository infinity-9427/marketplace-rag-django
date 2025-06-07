# RAG System

A Retrieval-Augmented Generation system for product information using Google's Gemini AI.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

3. Run the system:
```bash
python main.py
```

4. Run tests:
```bash
python main.py test
```

## Features

- Dynamic keyword extraction
- Intent detection
- Conversation history
- Multiple response types (brief, detailed, comparison)
- Product catalog search