from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Combines BM25 (keyword) + Semantic (vector) search
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Weight for semantic vs BM25 (0.5 = equal weight)
        """
        self.alpha = alpha
        self.bm25 = None
        self.chunk_texts = []
        self.chunk_ids = []
    
    def index_chunks(self, chunks: List[Dict]):
        """Build BM25 index from chunks"""
        
        self.chunk_texts = [c['text'] for c in chunks]
        self.chunk_ids = [c['id'] for c in chunks]
        
        # Tokenize for BM25
        tokenized = [text.lower().split() for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)
        
        logger.info(f"✅ Built BM25 index for {len(chunks)} chunks")
    
    def hybrid_search(
        self,
        query: str,
        semantic_results: List[Tuple[int, float]],  # (chunk_id, score)
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Combine BM25 + semantic scores
        
        Returns:
            List of (chunk_id, combined_score) tuples
        """
        
        if not self.bm25:
            logger.warning("BM25 not indexed, returning semantic only")
            return semantic_results[:k]
        
        # BM25 scores
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores to 0-1
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # Create score dict
        semantic_dict = {chunk_id: score for chunk_id, score in semantic_results}
        
        # Combine scores
        combined_scores = {}
        
        for i, chunk_id in enumerate(self.chunk_ids):
            semantic_score = semantic_dict.get(chunk_id, 0.0)
            bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0.0
            
            # Weighted combination
            combined = (
                self.alpha * semantic_score +
                (1 - self.alpha) * bm25_score
            )
            
            combined_scores[chunk_id] = combined
        
        # Sort and return top k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:k]