import sys
import json
from src.graph import app

def main():
    print("Welcome to the Policy RAG CLI. Type 'exit' to quit.")
    while True:
        question = input("\nAsk a question: ")
        if question.lower() in ["exit", "quit"]:
            break
        
        print("Thinking...")
        try:
            inputs = {"question": question}
            result = app.invoke(inputs)
            answer_data = result["answer"]
            
            print("\n--- Answer ---")
            print(answer_data["answer"])
            
            if answer_data.get("citations"):
                print("\n--- Citations ---")
                for cite in answer_data["citations"]:
                    print(f"- {cite}")
            
            print(f"\n[Confidence: {answer_data.get('confidence')}]")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
