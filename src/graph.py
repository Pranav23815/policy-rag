import os
import json
import re
from typing import TypedDict, List
from dotenv import load_dotenv

# LangChain Imports
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# Groq Import
from langchain_groq import ChatGroq

load_dotenv()

# --- CONFIGURATION ---
CHROMA_PATH = "./chroma_db"

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("CRITICAL: GROQ_API_KEY is missing from .env!")

# --- MODELS ---
print("Loading Embeddings...")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- LLM SETUP (THE FIX) ---
# We use the specific version that is currently live.
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # <--- THIS IS THE ONLY WORKING MODEL ID RIGHT NOW
    temperature=0,
    max_retries=2,
)

# --- GRAPH STATE ---
class GraphState(TypedDict):
    question: str
    docs: List[Document]
    answer: dict

# --- NODES ---

def retrieve(state: GraphState):
    print(f"--- Retrieving: {state['question']} ---")
    question = state["question"]
    # Retrieve top 10 for broad context
    retriever = db.as_retriever(search_kwargs={"k": 10})
    documents = retriever.invoke(question)
    return {"docs": documents}

def rerank(state: GraphState):
    print("--- Reranking ---")
    question = state["question"]
    documents = state["docs"]
    
    if not documents:
        return {"docs": []}
    
    # Score documents
    pairs = [[question, doc.page_content] for doc in documents]
    scores = reranker_model.predict(pairs)
    
    # Sort by score (High to Low)
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    # Keep top 3 best matches
    top_docs = [doc for doc, score in scored_docs[:3]]
    return {"docs": top_docs}

def generate(state: GraphState):
    print("--- Generating (Llama 3.3) ---")
    question = state["question"]
    documents = state["docs"]
    
    context_text = "\n\n".join([f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for doc in documents])
    
    # System Prompt
    system_msg = (
        "You are a helpful policy assistant. Answer purely based on the context provided. "
        "If the answer is missing, state 'I cannot find this information'. "
        "Return your answer as a valid JSON object with keys: 'answer', 'citations' (list of strings), and 'confidence' (High/Medium/Low). "
        "Do NOT format with markdown code blocks, just return the raw JSON string."
    )
    
    # Standard Chat Structure
    messages = [
        ("system", system_msg),
        ("human", f"Context:\n{context_text}\n\nQuestion: {question}")
    ]

    try:
        response = llm.invoke(messages)
        response_text = response.content
        
        # Robust Cleanup
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return {"answer": data}
        else:
            return {"answer": {"answer": response_text, "citations": [], "confidence": "Low"}}
            
    except Exception as e:
        print(f"[GROQ ERROR]: {e}")
        return {"answer": {"answer": f"Error: {str(e)}", "citations": [], "confidence": "Error"}}

# --- WORKFLOW ---
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("rerank", rerank)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()