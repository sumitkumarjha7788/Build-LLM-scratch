from rag import AdaptiveRAG

def test_rag():
    print("--- Testing Adaptive RAG ---")
    rag = AdaptiveRAG()
    
    # Add Documents
    rag.add_document("The capital of France is Paris.")
    rag.add_document("Photosynthesis is the process by which plants make food.")
    rag.add_document("The secret code for the vault is 9988.")
    rag.add_document("Machine learning requires data.")
    
    print(f"Knowledge Base: {len(rag.documents)} docs.")
    
    # Query 1: Fact Retrieval
    query = "What is the secret code?"
    print(f"\nQuery: {query}")
    docs, conf = rag.retrieve(query, top_k=1)
    print(f"Top Doc: {docs[0]}")
    print(f"Confidence: {conf:.4f}")
    
    if "9988" in docs[0]:
        print("Retrieval: PASSED (Found secret)")
    else:
        print("Retrieval: FAILED")

    # Query 2: Relevance Check (France)
    query2 = "capital France"
    print(f"\nQuery: {query2}")
    docs2, conf2 = rag.retrieve(query2, top_k=1)
    print(f"Top Doc: {docs2[0]}")
    
    if "Paris" in docs2[0]:
         print("Retrieval: PASSED (Found Paris)")
    else:
         print("Retrieval: FAILED")
         
    # Prompt Formatting
    params = rag.format_prompt(query, docs)
    print("\nFormatted Prompt Preview:")
    print(params)

if __name__ == '__main__':
    test_rag()
