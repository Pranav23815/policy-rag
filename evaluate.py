from src.graph import app
import json

def run_evaluation():
    test_cases = [
        {"type": "Answerable", "question": "What is the refund window for hardware?"},
        {"type": "Answerable", "question": "Are shipping duties included for international orders?"},
        {"type": "Edge Case", "question": "Can I get a refund after 31 days?"},
        {"type": "Unanswerable", "question": "What is the CEO's favorite color?"},
        {"type": "Unanswerable", "question": "Can I work from a coffee shop?"} # Tricky: policy says public wifi prohibited without VPN
    ]

    print(f"{'Type':<15} | {'Question':<40} | {'Answer Preview':<40}")
    print("-" * 100)

    for case in test_cases:
        inputs = {"question": case["question"]}
        result = app.invoke(inputs)
        answer_data = result["answer"]
        
        answer_text = answer_data["answer"]
        # Truncate for display
        preview = (answer_text[:75] + '..') if len(answer_text) > 75 else answer_text
        
        print(f"{case['type']:<15} | {case['question']:<40} | {preview}")
        
        # Uncomment to see full structured output
        # print(json.dumps(answer_data, indent=2))
        # print("\n")

if __name__ == "__main__":
    run_evaluation()
