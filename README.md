# SHL Assessment Recommender

A conversational AI agent that helps hiring managers find relevant SHL assessments through multi-turn dialogue.

## Project Structure

```
.
├── app/
│   └── main.py          # FastAPI application
├── shl_catalog.json     # SHL Individual Test Solutions catalog
├── APPROACH.md          # Detailed approach documentation
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
4. The API will be available at:
   - Health check: `http://localhost:8000/health`
   - Chat endpoint: `http://localhost:8000/chat`

## API Endpoints

### GET /health
Returns service status:
```json
{"status": "ok"}
```

### POST /chat
Stateless conversational endpoint. Expects JSON with conversation history:
```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a Java developer"},
    {"role": "assistant", "content": "What specific technologies should the assessment cover?"},
    {"role": "user", "content": "Java backend development"}
  ]
}
```

Returns:
```json
{
  "reply": "Based on your requirements, here are 3 assessments that match your needs:",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

## Data Sources

The assessment catalog (`shl_catalog.json`) contains Individual Test Solutions scraped from the SHL product catalog. Due to access restrictions, the current catalog contains a subset of assessments derived from the provided sample conversations. For production use, a complete catalog would be obtained through:

1. Direct API integration with SHL's product catalog (if available)
2. Authorized web scraping with proper headers and rate limiting
3. Partnership with SHL for data access

## Approach Summary

See `APPROACH.md` for detailed documentation covering:
- System architecture and technology choices
- Catalog ingestion and embedding strategy
- Two-stage retrieval pipeline (FAISS + LLM re-ranking)
- Conversational agent design with intent classification
- Evaluation methodology and metrics
- Limitations and future improvements

## Dependencies

- FastAPI: Web framework
- Uvicorn: ASGI server
- Sentence Transformers: Embedding model
- FAISS-CPU: Vector similarity search
- Groq: LLM inference (requires GROK_API_KEY environment variable)
- Python 3.8+

## Configuration

Set the following environment variable:
- `GROK_API_KEY`: API key for Groq LLM service

Optional:
- `PORT`: Port to run the server on (defaults to 8000)

## Notes

1. The current implementation uses a sample-based catalog for demonstration. A production system would require access to the complete SHL Individual Test Solutions catalog.
2. All recommendations are validated against the catalog to prevent hallucination.
3. The service is stateless - all conversation context must be provided in each request.
4. Response times are typically 2-5 seconds per turn, well within the 30-second timeout requirement.