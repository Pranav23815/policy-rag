import os
import shutil
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths
DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"

def load_documents():
    """Loads text files from the data directory."""
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents

def split_text(documents: List):
    """
    Splits documents into smaller chunks.
    
    Why chunk_size=500 and chunk_overlap=50?
    - Size 500: Small enough to capture specific policy clauses (e.g., "30-day refund window") without dragging in unrelated context from other sections.
    - Overlap 50: Ensures that context isn't lost if a sentence is split right in the middle of a keyword or phrase.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: List):
    """Saves the chunks to a local ChromaDB instance."""
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Use HuggingFace embeddings (runs locally, free)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def main():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

if __name__ == "__main__":
    main()
