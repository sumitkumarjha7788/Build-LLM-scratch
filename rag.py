import torch

class AdaptiveRAG:
    """
    Adaptive Retrieval-Augmented Generation module.
    Simulates a knowledge base and hybrid search.
    """
    def __init__(self, documents=None):
        # In production, this would be a Vector DB (FAISS/Chroma)
        self.documents = documents if documents else []
        
    def add_document(self, text):
        self.documents.append(text)
        
    def retrieve(self, query: str, top_k: int = 2):
        """
        Retrieves top_k relevant documents.
        Uses a mock Hybrid Search (Keyword + Semantic).
        """
        scores = []
        for i, doc in enumerate(self.documents):
            # 1. Keyword Score (BM25 approximation)
            # Simple overlap count
            query_terms = set(query.lower().split())
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            keyword_score = overlap / (len(query_terms) + 1e-6)
            
            # 2. Semantic Score (Mock)
            # In real system: cos_sim(embed(query), embed(doc))
            # Here: just random or length based proxy for demo
            semantic_score = 0.5 # Dummy neutral score
            
            # Weighted sum
            hybrid_score = 0.7 * keyword_score + 0.3 * semantic_score
            scores.append((hybrid_score, doc))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Adaptive Logic: Check confidence of top 1
        top_docs = [doc for score, doc in scores[:top_k]]
        confidence = scores[0][0] if scores else 0
        
        return top_docs, confidence

    def format_prompt(self, query, retrieved_docs):
        """ Concatenates docs and query into a RAG prompt """
        context_str = "\n\n".join([f"Doc {i+1}: {d}" for i, d in enumerate(retrieved_docs)])
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        return prompt
