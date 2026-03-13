def build_rag_prompt(query: str, context: str) -> str:
    """
    Build RAG prompt with context
    
    Args:
        query: User question
        context: Retrieved document chunks
        
    Returns:
        Formatted prompt for LLM
    """
    
    prompt = f"""You are a helpful AI assistant answering questions based on provided document context.

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question using ONLY information from the context above
2. If the context doesn't contain enough information, say "I don't have enough information to answer that question"
3. Be concise and direct
4. Cite which parts of the context you used (e.g., "According to the document...")
5. Do not make up information

ANSWER:"""
    
    return prompt


def build_simple_prompt(query: str) -> str:
    """
    Build a simple prompt without context (for testing)
    """
    
    prompt = f"""You are a helpful AI assistant.

Question: {query}

Answer the question clearly and concisely.

Answer:"""
    
    return prompt


RAG_SYSTEM_PROMPT = """You are an intelligent document assistant. Your role is to answer questions accurately based on the provided document context. Always cite your sources and admit when you don't have enough information."""