Policy RAG CLI Tool
Overview This is a production-grade Retrieval-Augmented Generation (RAG) tool. It allows employees to ask questions about company policies (Refunds, Shipping, Remote Work) and get accurate answers backed by citations. It uses Llama 3.3 (via Groq) for inference and ChromaDB for retrieval.

Features

Retrieval: Uses ChromaDB to find relevant policy documents.

Reranking: Uses a Cross-Encoder to filter results, ensuring high accuracy.

Anti-Hallucination: The system refuses to answer if the information is missing.

Structured Output: Returns answers in JSON format with confidence scores.

Setup Instructions

Step 1: Install Dependencies Run the following command in your terminal: pip install -r requirements.txt

Step 2: Set up Environment Create a file named ".env" in your folder. Add your API key inside it like this: GROQ_API_KEY="your_actual_key_here"

How to Run the Project

1. Ingest Data (Prepare the Database) Run this command to load the policy files: python src/ingest.py

2. Run Evaluation (Test the System) Run this command to check for accuracy: python evaluate.py

3. Start the Chat (Ask Questions) Run this command to start the tool: python main.py

Architecture Overview

1. Data Ingestion We split the policy documents into small chunks (500 characters). This size was chosen because it captures single rules (like "30-day refund window") without mixing up different topics.

2. Retrieval When you ask a question, the system searches ChromaDB for the top 10 most similar text chunks.

3. Reranking We use a "Cross-Encoder" model to double-check those 10 chunks. It reads them carefully and picks the Top 3 that actually answer the question. This is much smarter than standard search.

4. Generation We send the Top 3 chunks to Llama 3.3. The system prompt forces the AI to be honest: if the answer isn't there, it must say "I cannot find this information."

### Visual Workflow
```mermaid
graph LR
    A[User Query] --> B(Retrieve Node)
    B -->|Top 10 Docs| C(Rerank Node)
    C -->|Top 3 Docs| D(Generate Node)
    D --> E[Final Answer]
    
    subgraph Data Pipeline
    F[PDF/TXT Policies] -->|Ingest| G[ChromaDB]
    end
    
    G -.->|Search| B

Prompt Engineering
Initial Attempt: We started with a simple prompt: "Answer the question based on the text."

Result: It hallucinated and gave vague answers.

Final Version: We switched to a strict prompt: "You are a policy assistant. Answer ONLY using the provided context. If the answer is missing, say 'I cannot find this information'. Return the answer in JSON format."

Result: Zero hallucinations and clear citations.

Trade-offs & Reflections

Why Groq & Llama 3.3? It is extremely fast and free to use, though it has strict rate limits compared to OpenAI.

Why Reranking? It adds a small delay (latency), but it drastically improves accuracy. Without it, the system might miss subtle answers.

What I am most proud of: Implementing the Reranking step. It stops the AI from guessing and forces it to select the absolute best evidence before answering.

What I would improve next: I would implement "Hybrid Search" (combining keyword search with vector search) to handle exact ID lookups better.