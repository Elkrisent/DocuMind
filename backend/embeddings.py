from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging
import torch

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generate embeddings for text chunks and image captions
    Uses sentence-transformers for semantic similarity
    """
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model identifier
                       all-MiniLM-L6-v2: 384 dim, fast, good quality
        """
        logger.info(f"🔄 Loading embedding model: {model_name}")
        
        # Load model
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Use GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("✅ Using GPU for embeddings")
        else:
            logger.info("✅ Using CPU for embeddings")
        
        logger.info(f"✅ Model loaded: {self.embedding_dim}-dimensional embeddings")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Generate embedding
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        return embedding.astype(np.float32)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts (more efficient)
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        # Filter out empty texts but keep track of indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            # All texts were empty
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        logger.info(f"🔄 Embedding {len(valid_texts)} texts in batches of {batch_size}")
        
        # Generate embeddings
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # Create result array with zeros for empty texts
        result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for i, valid_idx in enumerate(valid_indices):
            result[valid_idx] = embeddings[i]
        
        return result
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query
        Same as embed_text but explicit for clarity
        
        Args:
            query: Search query string
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.embed_text(query)


# Global singleton instance
_embedding_generator = None

def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create the global embedding generator instance"""
    global _embedding_generator
    
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    
    return _embedding_generator


async def generate_chunk_embeddings(chunks: List[dict]) -> np.ndarray:
    """
    Generate embeddings for a list of chunks
    
    Args:
        chunks: List of chunk dicts with 'text' field
        
    Returns:
        numpy array of embeddings
    """
    generator = get_embedding_generator()
    
    # Extract texts
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings
    embeddings = generator.embed_batch(texts)
    
    logger.info(f"✅ Generated {len(embeddings)} chunk embeddings")
    
    return embeddings


async def generate_caption_embeddings(captions: List[str]) -> np.ndarray:
    """
    Generate embeddings for image captions
    
    Args:
        captions: List of caption strings
        
    Returns:
        numpy array of embeddings
    """
    generator = get_embedding_generator()
    
    # Generate embeddings
    embeddings = generator.embed_batch(captions)
    
    logger.info(f"✅ Generated {len(embeddings)} caption embeddings")
    
    return embeddings