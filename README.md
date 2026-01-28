# Policy RAG CLI Tool

A production-grade Retrieval-Augmented Generation (RAG) CLI tool for answering questions about company policies.

## Features
- **Retrieval:** Uses ChromaDB with standard cosine similarity.
- **Reranking:** Sorts retrieved documents using `CrossEncoder` for higher relevance.
- **Orchestration:** LangGraph (StateGraph) manages the pipeline flow.
- **Validation:** Pydantic ensures structured, valid JSON output.
- **Anti-Hallucination:** Strict prompts enforce honesty.

## Setup Instructions

1.  **Clone & Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Setup**
    Copy the example env file and add your API key:
    ```bash
    cp .env.example .env
    # Edit .env and set OPENAI_API_KEY=sk-...
    ```

## Architecture Overview

The system processes a user query through the following LangGraph nodes:

1.  **Retrieve (`Node 1`):**
    -   Fetches top `k=10` documents from the ChromaDB vector store.
    -   Model: `all-MiniLM-L6-v2` (HuggingFace).

2.  **Rerank (`Node 2`):**
    -   Scores the 10 documents against the query using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
    -   Filters down to the top `k=3` highest-scoring documents.
    -   **Benefit:** Improves precision by analyzing the query-document pair deeply, which vector similarity alone might miss.

3.  **Generate (`Node 3`):**
    -   LLM: `gpt-4o-mini` (OpenAI).
    -   Uses a strict system prompt and enforces JSON output via Pydantic.

## Prompt Iteration: Naive vs. Structured

We evolved the prompt strategy to ensure reliability:

### Version 1: Naive (Unused)
> "Answer the question based on the text."

*Why it fails:* It allows the model to seamlessly blend external knowledge with the context, leading to "plausible" hallucinations or answering out-of-policy questions (e.g., general knowledge) which is undesirable for a compliance tool.

### Version 2: Structured (Final)
> "You are a helpful assistant for a company policy handbook. Answer ONLY using the provided context. If the answer is not in the context, strictly say 'I cannot find this information in the policy.' Do not hallucinate."

*Why we use it:*
1.  **Constraint:** Forces the model to admit ignorance ("I cannot find...") rather than guessing.
2.  **Structure:** Helper text instructs it to act as a policy handbook assistant.

## Trade-offs

| Feature | Pros | Cons |
| :--- | :--- | :--- |
| **Reranking** | High retrieval accuracy; surfaces semantic matches better than simple vector search. | Adds latency (running a BERT model on 10 docs takes time); requires more compute. |
| **Pydantic** | Guarantees specific JSON structure for frontend integration. | Slightly higher token cost; strict validation might fail if model outputs slightly malformed JSON (rare with GPT-4). |
| **Small Chunks (500)** | Precise citations; less irrelevant noise. | May lose context if a rule spans multiple paragraphs (mitigated by overlap). |

## Execution Guide

**1. Create Environment & Install**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**2. Ingest Data (Prepare DB)**
```bash
python -m src.ingest
```

**3. Run Evaluation (Test Cases)**
```bash
python evaluate.py
```

**4. Run CLI (Interactive)**
```bash
python main.py
```
