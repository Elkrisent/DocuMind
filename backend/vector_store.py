import faiss
import numpy as np
from pathlib import Path
import pickle
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    FAISS-based vector store for semantic search
    Supports both L2 and cosine similarity
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        index_type: str = "flat",
        metric: str = "cosine"
    ):
        """
        Initialize FAISS index
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index ("flat" for exact search)
            metric: Distance metric ("cosine" or "l2")
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        
        # Create index
        if metric == "cosine":
            # For cosine similarity, use inner product with normalized vectors
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            # L2 distance
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Metadata storage (maps index position → chunk/caption ID)
        self.id_map = []  # List of (type, id) tuples
        # type: "chunk" or "caption"
        # id: database ID
        
        logger.info(f"✅ Created FAISS index: {index_type}, {metric} metric, {embedding_dim}D")
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        ids: List[int],
        data_type: str = "chunk"
    ):
        """
        Add embeddings to the index
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            ids: List of database IDs corresponding to embeddings
            data_type: Type of data ("chunk" or "caption")
        """
        if len(embeddings) == 0:
            logger.warning("⚠️  No embeddings to add")
            return
        
        # Ensure correct shape
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Verify dimensions
        assert embeddings.shape[1] == self.embedding_dim, \
            f"Embedding dimension mismatch: {embeddings.shape[1]} vs {self.embedding_dim}"
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Update ID map
        for id_val in ids:
            self.id_map.append((data_type, id_val))
        
        logger.info(f"✅ Added {len(embeddings)} {data_type} embeddings to index")
        logger.info(f"📊 Total vectors in index: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, int, float]]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            k: Number of results to return
            
        Returns:
            List of (data_type, id, score) tuples
            Sorted by relevance (highest score first)
        """
        if self.index.ntotal == 0:
            logger.warning("⚠️  Index is empty")
            return []
        
        # Ensure correct shape
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Limit k to available vectors
        k = min(k, self.index.ntotal)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert to results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.id_map):  # Valid index
                data_type, id_val = self.id_map[idx]
                score = float(distances[0][i])
                
                # Convert distance to similarity score
                # For cosine (IP): higher is better (already similarity)
                # For L2: lower is better (convert to similarity)
                if self.metric == "l2":
                    score = 1 / (1 + score)  # Convert distance to similarity
                
                results.append((data_type, id_val, score))
        
        return results
    
    def save(self, index_path: str, metadata_path: str):
        """Save index and metadata to disk"""
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            'id_map': self.id_map,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"💾 Saved index to {index_path}")
        logger.info(f"💾 Saved metadata to {metadata_path}")
    
    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> 'FAISSVectorStore':
        """Load index and metadata from disk"""
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(
            embedding_dim=metadata['embedding_dim'],
            index_type=metadata['index_type'],
            metric=metadata['metric']
        )
        
        # Load FAISS index
        instance.index = faiss.read_index(index_path)
        instance.id_map = metadata['id_map']
        
        logger.info(f"📂 Loaded index from {index_path}")
        logger.info(f"📊 Index contains {instance.index.ntotal} vectors")
        
        return instance
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        chunk_count = sum(1 for t, _ in self.id_map if t == "chunk")
        caption_count = sum(1 for t, _ in self.id_map if t == "caption")
        
        return {
            'total_vectors': self.index.ntotal,
            'chunk_vectors': chunk_count,
            'caption_vectors': caption_count,
            'embedding_dim': self.embedding_dim,
            'metric': self.metric
        }


# Global singleton instance
_vector_store = None
_index_path = Path("/documents/faiss_index.bin")
_metadata_path = Path("/documents/faiss_metadata.pkl")

def get_vector_store() -> FAISSVectorStore:
    """Get or create the global vector store instance"""
    global _vector_store
    
    if _vector_store is None:
        # Try to load existing index
        if _index_path.exists() and _metadata_path.exists():
            try:
                _vector_store = FAISSVectorStore.load(
                    str(_index_path),
                    str(_metadata_path)
                )
                logger.info("✅ Loaded existing FAISS index")
            except Exception as e:
                logger.warning(f"⚠️  Could not load existing index: {e}")
                _vector_store = FAISSVectorStore()
        else:
            _vector_store = FAISSVectorStore()
    
    return _vector_store

def save_vector_store():
    """Save the global vector store to disk"""
    if _vector_store is not None:
        _vector_store.save(str(_index_path), str(_metadata_path))