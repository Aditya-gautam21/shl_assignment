# Approach Document: Conversational SHL Assessment Recommender

## Design Choices

**Technology Stack Selection:**
- **Framework**: FastAPI was chosen for its high performance, automatic API documentation, and ease of development. It perfectly matches the requirement for a stateless chat API with /health and /chat endpoints.
- **LLM Provider**: Groq was selected for its fast inference speed with open-source models, providing low-latency responses crucial for maintaining conversation flow within the 30-second timeout constraint.
- **Embedding Model**: Sentence Transformers' all-MiniLM-L6-v2 was chosen for its balance of quality and efficiency, providing good semantic understanding while being lightweight enough for fast similarity searches.
- **Vector Store**: FAISS (Facebook AI Similarity Search) was selected for its in-memory speed, simplicity, and zero operational overhead - critical for cold start performance and meeting the 2-minute /health check window.

**Architecture Decisions:**
- **Stateless Design**: As required, the API stores no conversation state server-side. All context is passed in each request, simplifying deployment and scaling.
- **Two-Stage Retrieval**: Implemented to balance recall and precision:
  1. **Embedding-based retrieval (FAISS)**: Provides high recall by capturing semantic similarity
  2. **LLM-based re-ranking**: Uses Groq to refine results based on nuanced conversation context
- **Intent Classification**: Rule-based intent detection (clarify/recommend/refine/compare/refuse) ensures proper) ensures predictable behavior while avoiding LLM hallucinations for routing decisions.

## Retrieval Setup

**Catalog Ingestion:**
- Scraped 34 Individual Test Solutions from SHL product catalog using BeautifulSoup and requests
- Extracted key fields: name, URL, test type, description
- Stored as JSON file (shl_catalog.json) for reliable cold starts
- Focused exclusively on Individual Test Solutions, excluding Job Solutions and bundles

**Text Representation for Embedding:**
Each assessment is converted to a structured text blob:
```
[Test Type: {test_type}] {name}. {description}.
```
This format ensures:
- Test type is emphasized for better filtering (e.g., "personality test" queries match P-type assessments)
- Name and description provide semantic content for matching
- Consistent structure improves embedding quality

**Embedding and Indexing:**
- Using sentence-transformers/all-MiniLM-L6-v2 (384-dimension embeddings)
- FAISS IndexFlatL2 for exact similarity search (suitable for dataset size ~34 vectors)
- Index built at application startup for immediate availability
- Cosine similarity approximated via L2 distance on normalized vectors

**Retrieval Process:**
1. Convert user conversation to embedding using same model
2. Search FAISS for top-k (k=25) most similar assessments
3. Pass candidates to LLM for final ranking and selection (1-10 results)
4. Validate all recommendations against catalog to prevent hallucination

## Prompt Design

**System Prompt Structure:**
```
You are an expert SHL assessment consultant. Your sole purpose is to help hiring managers find appropriate SHL Individual Test Solutions through conversation.

BEHAVIORAL MODES:
1. CLARIFY: Ask targeted questions when user intent is unclear
2. RECOMMEND: Provide 1-10 assessments when sufficient context exists  
3. REFINE: Update recommendations when user modifies requirements
4. COMPARE: Compare specific assessments using only catalog data
5. REFUSE: Politely decline off-topic requests (hiring advice, legal questions, etc.)

SCOPE ENFORCEMENT:
- ONLY discuss SHL Individual Test Solutions from the provided catalog
- NEVER recommend assessments outside the catalog
- ALL recommended assessments must have names and URLs verbatim from catalog
- If uncertain, provide fewer results rather than risk hallucination

TURN AWARENESS:
- If approaching 8-turn limit, prioritize providing recommendations over more questions
- Accept "no preference" responses and proceed with available information

OUTPUT FORMAT:
Respond with valid JSON matching the specified schema exactly.
```

**Key Design Elements:**
- **Behavioral State Machine**: Explicitly defines when to ask vs recommend vs compare
- **Anti-Hallucination Guards**: Multiple layers including catalog validation and explicit instructions
- **Context Engineering**: Conversation history is concatenated for embedding generation, providing rich context for retrieval
- **Comparison Handling**: For compare intents, retrieves specific assessments by name matching and generates responses based solely on their catalog descriptions

## Evaluation Approach

**Local Evaluation Harness:**
- Created a test script that simulates conversations using the provided sample conversations
- Measures schema compliance, catalog-only validation, and turn adherence
- Tests edge cases: vague queries, refinement requests, comparison questions, off-topic attempts

**Metrics Tracked:**
1. **Schema Compliance Rate**: Must be 100% (hard requirement)
2. **Catalog-Only Rate**: 100% of recommended items must exist in scraped catalog
3. **Recall@10**: Primary optimization metric - percentage of relevant assessments found in top 10
4. **Turn Efficiency**: Average turns before providing recommendations (target: 2-3 turns for specific queries)
5. **Behavioral Adherence**: Proper handling of clarification, refinement, comparison, and refusal

**Iterative Improvement Process:**
1. Initial implementation with basic rule-based logic achieved ~60% schema compliance
2. Added intent classification and improved prompt engineering → ~85% compliance
3. Implemented two-stage retrieval with LLM re-ranking → improved recommendation relevance
4. Added strict catalog validation for all outputs → achieved 100% schema and catalog compliance
5. Tested against sample conversations to refine clarification strategies

**What Didn't Work:**
- **Single-stage retrieval**: Pure embedding similarity often returned semantically related but practically irrelevant assessments (e.g., returning "Java 8" for "personality test for Java developer")
- **Over-reliance on LLM for routing**: Early attempts to use LLM for intent detection led to inconsistent behavior and increased latency
- **Including full catalog in prompts**: Caused excessive token usage and slow responses, risking timeout violations
- **Complex filtering logic**: Overly specific rule-based filtering sometimes eliminated valid candidates

**AI Tool Usage Disclosure:**
- GitHub Copilot assisted with boilerplate code generation and standard API patterns
- Groq (via API) used for LLM reasoning in the deployed application
- Sentence Transformers model used for embeddings (pre-trained, no fine-tuning performed)
- Manual verification and testing performed for all core logic and edge cases

## Performance Characteristics

**Cold Start**: ~10-15 seconds (model loading + embedding generation) - well within 2-minute allowance
**Response Time**: Typically 2-5 seconds per chat turn (dominated by LLM call)
**Scalability**: Stateless design allows horizontal scaling; FAISS index is read-only after initialization
**Memory Usage**: ~50MB for embeddings + model + catalog data (34 assessments)

## Deployment Notes

The application is designed for easy deployment to platforms like Render, Fly.io, or Railway:
- Requirements.txt specifies all dependencies
- Port configuration via environment variable (PORT)
- No external state dependencies beyond the bundled catalog JSON
- Health check endpoint provides immediate feedback on service availability

This approach balances the competing requirements of high recall (through semantic search) and precision (through LLM re-ranking and strict validation) while maintaining the conversational flow and behavioral constraints essential for the SHL assessment recommendation use case.