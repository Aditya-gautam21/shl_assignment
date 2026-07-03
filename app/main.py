import os
import json
import re
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from sentence_transformers import SentenceTransformer
import logging
from groq import Groq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROK_API_KEY"))

# Initialize FastAPI app
app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")

# CORS origins from env (comma-separated), fallback to localhost for dev
cors_origins = os.environ.get("CORS_ORIGINS", "").split(",")
if not cors_origins or cors_origins == [""]:
    cors_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

# Global variables for the retrieval system
catalog_data = []
embeddings = None
faiss_index = None
embedder = None

# Load catalog and initialize retrieval system on startup
@app.on_event("startup")
async def startup_event():
    global catalog_data, embeddings, faiss_index, embedder
    logger.info("Loading catalog data...")
    catalog_data = load_catalog()
    
    logger.info("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    logger.info("Creating embeddings...")
    embeddings = create_embeddings(catalog_data)
    
    logger.info("Building FAISS index...")
    global faiss_index
    faiss_index = build_faiss_index(embeddings)
    
    logger.info("Startup complete!")

def is_individual_test_solution(assessment: Dict[str, Any]) -> bool:
    """
    Check if the assessment is an Individual Test Solution (not a Job Solution or bundle).
    """
    name = assessment.get('name', '').lower()
    description = assessment.get('description', '').lower()
    
    # Exclude known job solution/bundle indicators
    exclude_keywords = [
        'job fit', 'job matching', 'candidate report', 'development report',
        '360', 'feedback', 'inventory', 'questionnaire', 'assessment center',
        'job simulation', 'role play', 'case study', 'interview guide',
        'selection report', 'development report', 'report', 'catalog', 'bundle', 'solution'
    ]
    
    for keyword in exclude_keywords:
        if keyword in name or keyword in description:
            return False
    
    # Check URL - individual assessments usually have /view/ in the path
    url = assessment.get('url', '').lower()
    if '/view/' in url:
        return True
    
    # If we have a test type classification, it's likely an individual test
    if assessment.get('test_type') in ['K', 'P', 'A', 'S', 'B', 'C', 'D', 'E']:
        return True
    
    # Default to True if we can't determine it's a job solution (conservative)
    return True


def load_catalog() -> List[Dict[str, Any]]:
    """Load the SHL catalog from JSON file and clean the data."""
    catalog_path = "shl_catalog.json"
    if not os.path.exists(catalog_path):
        logger.warning(f"Catalog file {catalog_path} not found. Creating empty catalog.")
        return []
    
    with open(catalog_path, 'r') as f:
        data = json.load(f)
    
    cleaned_data = []
    for assessment in data:
        # Clean URL: remove angle brackets if present
        url = assessment.get('url', '')
        if url.startswith('<') and url.endswith('>'):
            url = url[1:-1]
        
        # Clean test_type: take first letter if multiple, ensure uppercase, default to 'U'
        test_type = assessment.get('test_type', '')
        if test_type:
            # Take the first alphabetic character
            match = re.search(r'[A-Za-z]', test_type)
            if match:
                test_type = match.group(0).upper()
                # Only keep if it's one of the expected types, otherwise default to 'U'
                if test_type not in ['K', 'P', 'A', 'S', 'B', 'C', 'D', 'E']:
                    test_type = 'U'
            else:
                test_type = 'U'
        else:
            test_type = 'U'
        
        # Create a temporary assessment with cleaned test_type for the individual test check
        temp_assessment = {
            'name': assessment.get('name', ''),
            'url': url,
            'test_type': test_type,
            'description': assessment.get('description', '')
        }
        
        # Only include if it's an individual test solution
        if is_individual_test_solution(temp_assessment):
            cleaned_assessment = {
                'name': assessment.get('name', 'Unknown Assessment'),
                'url': url,
                'test_type': test_type,
                'description': assessment.get('description', '')
            }
            cleaned_data.append(cleaned_assessment)
    
    logger.info(f"Loaded and cleaned {len(cleaned_data)} individual test solutions")
    return cleaned_data

def create_text_for_embedding(assessment: Dict[str, Any]) -> str:
    """Create a text representation of an assessment for embedding."""
    parts = []
    
    if assessment.get('name'):
        parts.append(f"Name: {assessment['name']}")
    
    if assessment.get('test_type'):
        parts.append(f"Type: {assessment['test_type']}")
    
    if assessment.get('description'):
        parts.append(f"Description: {assessment['description']}")
    
    return ". ".join(parts)

def create_embeddings(catalog_data: List[Dict[str, Any]]) -> np.ndarray:
    """Create embeddings for all assessments in the catalog."""
    if not catalog_data:
        return np.array([])
    
    texts = [create_text_for_embedding(item) for item in catalog_data]
    embeddings = embedder.encode(texts)
    return embeddings.astype('float32')

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS index from embeddings."""
    if len(embeddings) == 0:
        # Return a dummy index if no embeddings
        dimension = 384  # Default dimension for all-MiniLM-L6-v2
        index = faiss.IndexFlatL2(dimension)
        return index
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_similar_assessments(query: str, k: int = 10) -> List[int]:
    """Search for similar assessments using FAISS."""
    if len(catalog_data) == 0 or faiss_index.ntotal == 0:
        return []
    
    query_embedding = embedder.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, k)
    return indices[0].tolist()

def classify_intent_with_llm(messages: List[Message]) -> str:
    """Use LLM to classify the intent of the conversation."""
    try:
        # Prepare conversation history
        conversation = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        
        prompt = f"""Classify the user's intent in this conversation for an SHL assessment recommender system.

Conversation:
{conversation}

Possible intents:
1. CLARIFY - The user's query is too vague and needs clarification
2. RECOMMEND - The user has provided enough information to recommend assessments
3. REFINE - The user is modifying or refining previous requirements
4. COMPARE - The user wants to compare specific assessments
5. REFUSE - The request is off-topic or not about SHL assessments

Respond with only the intent name (CLARIFY, RECOMMEND, REFINE, COMPARE, or REFUSE)."""

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=10,
        )
        
        intent = chat_completion.choices[0].message.content.strip().upper()
        
        # Validate the intent
        if intent in ["CLARIFY", "RECOMMEND", "REFINE", "COMPARE", "REFUSE"]:
            return intent
        else:
            # Default to CLARIFY if invalid response
            return "CLARIFY"
    except Exception as e:
        logger.error(f"Error in LLM intent classification: {e}")
        # Fallback to rule-based classification
        return classify_intent(messages)

def classify_intent(messages: List[Message]) -> str:
    """Fallback rule-based intent classification."""
    if not messages:
        return "CLARIFY"
    
    last_message = messages[-1].content.lower()
    
    # Check for comparison requests
    if any(word in last_message for word in ["difference", "compare", "vs", "versus", "better"]):
        return "COMPARE"
    
    # Check for refinement requests
    if any(word in last_message for word in ["add", "remove", "instead", "rather", "but", "however"]):
        return "REFINE"
    
    # Check if we have enough information to recommend
    skills_keywords = [
        "java", "python", "javascript", "developer", "engineer", "manager",
        "sales", "customer service", "leadership", "personality", "cognitive",
        "numerical", "verbal", "logical", "technical"
    ]
    
    has_skill_or_role = any(keyword in last_message for keyword in skills_keywords)
    
    # If we have several messages and mention skills/roles, we can recommend
    if len(messages) >= 2 and has_skill_or_role:
        return "RECOMMEND"
    
    # Otherwise, we need to clarify
    return "CLARIFY"

def generate_clarifying_question(messages: List[Message]) -> str:
    """Generate a clarifying question based on conversation history."""
    if not messages:
        return "Hello! I'd be happy to help you find the right SHL assessment. Could you tell me what role or skills you're looking to assess?"
    
    last_message = messages[-1].content.lower()
    
    # If we have some info but need more specifics
    if any(word in last_message for word in ["developer", "engineer", "programmer"]):
        return "What specific programming languages or technologies should the assessment cover?"
    elif any(word in last_message for word in ["manager", "leader", "supervisor"]):
        return "What leadership or management competencies are you looking to assess?"
    elif any(word in last_message for word in ["sales", "customer service"]):
        return "What specific sales or customer service skills are important for this role?"
    else:
        return "Could you please tell me more about the role you're hiring for and what specific skills or competencies you want to assess?"

def generate_recommendation_reply(messages: List[Message], recommendations: List[Dict]) -> str:
    """Generate a reply when providing recommendations."""
    if not recommendations:
        return "I'm sorry, I couldn't find any assessments that match your criteria. Could you provide more details about what you're looking for?"
    
    count = len(recommendations)
    if count == 1:
        return "Based on your requirements, I recommend the following assessment:"
    else:
        return f"Based on your requirements, here are {count} assessments that match your needs:"

def generate_comparison_reply(messages: List[Message], assessments: List[Dict]) -> str:
    """Generate a reply when comparing assessments."""
    if len(assessments) < 2:
        return "I need at least two assessments to compare. Could you specify which assessments you'd like to compare?"
    
    names = [a['name'] for a in assessments]
    return f"Here's a comparison of {', '.join(names)}:"

def generate_refinement_reply(messages: List[Message], recommendations: List[Dict]) -> str:
    """Generate a reply when refining recommendations."""
    if not recommendations:
        return "I've updated the recommendations based on your new criteria, but I couldn't find any matching assessments. Could you adjust your requirements?"
    
    count = len(recommendations)
    return f"I've refined the recommendations based on your updated criteria. Here are {count} assessments that match your new requirements:"

def generate_refusal_reply() -> str:
    """Generate a refusal reply for off-topic requests."""
    return "I'm sorry, but I can only help with SHL assessments. Please ask me about assessment selection, comparison, or refinement for hiring purposes."

def format_recommendations(assessments: List[Dict]) -> List[Recommendation]:
    """Format assessment data for the response."""
    recommendations = []
    for assessment in assessments:
        recommendations.append(Recommendation(
            name=assessment.get('name', 'Unknown Assessment'),
            url=assessment.get('url', '#'),
            test_type=assessment.get('test_type', 'U')
        ))
    return recommendations

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for the SHL Assessment Recommender."""
    try:
        # Classify the intent using LLM
        intent = classify_intent_with_llm(request.messages)
        logger.info(f"Classified intent: {intent}")
        
        # Handle different intents
        if intent == "REFUSE":
            return ChatResponse(
                reply=generate_refusal_reply(),
                recommendations=[],
                end_of_conversation=False
            )
        
        elif intent == "COMPARE":
            # Extract assessment names from the last message for comparison
            # This is a simplified implementation
            last_message = request.messages[-1].content
            # In a real implementation, we'd use NER or more sophisticated extraction
            return ChatResponse(
                reply=generate_comparison_reply(request.messages, []),
                recommendations=[],
                end_of_conversation=False
            )
        
        elif intent == "REFINE":
            # For refinement, we'd need to extract new constraints and re-run search
            # For now, we'll do a basic search on the last message
            last_message = request.messages[-1].content
            indices = search_similar_assessments(last_message, k=10)
            recommended_assessments = [catalog_data[i] for i in indices if i < len(catalog_data)]
            
            return ChatResponse(
                reply=generate_refinement_reply(request.messages, recommended_assessments),
                recommendations=format_recommendations(recommended_assessments),
                end_of_conversation=False
            )
        
        elif intent == "RECOMMEND":
            # Get recommendations based on conversation history
            # Combine all user messages for context
            user_messages = [msg.content for msg in request.messages if msg.role == "user"]
            conversation_text = " ".join(user_messages)
            indices = search_similar_assessments(conversation_text, k=10)
            recommended_assessments = [catalog_data[i] for i in indices if i < len(catalog_data)]
            
            return ChatResponse(
                reply=generate_recommendation_reply(request.messages, recommended_assessments),
                recommendations=format_recommendations(recommended_assessments),
                end_of_conversation=False
            )
        
        else:  # CLARIFY
            return ChatResponse(
                reply=generate_clarifying_question(request.messages),
                recommendations=[],
                end_of_conversation=False
            )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)