from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

# Request/Response schemas for API

class DocumentCreate(BaseModel):
    """Schema for creating a document"""
    filename: str

class DocumentResponse(BaseModel):
    """Schema for document response"""
    id: int
    filename: str
    original_filename: str
    file_size: int
    status: str
    num_pages: Optional[int] = None
    num_chunks: int
    num_images: int = 0  # Add this line
    has_ocr: bool = False  # Add this line
    uploaded_at: datetime
    category: Optional[str]
    keywords: Optional[str]
    
    class Config:
        from_attributes = True

class ChunkResponse(BaseModel):
    """Schema for chunk response"""
    id: int
    chunk_index: int
    text: str
    page_num: Optional[int] = None
    
    class Config:
        from_attributes = True

class DocumentDetailResponse(DocumentResponse):
    """Document with its chunks"""
    chunks: List[ChunkResponse] = []

class QueryCreate(BaseModel):
    """Schema for creating a query"""
    query_text: str = Field(..., min_length=1, max_length=1000)

class QueryResponse(BaseModel):
    """Schema for query response"""
    id: int
    query_text: str
    latency_ms: Optional[int] = None
    cache_hit: bool
    timestamp: datetime
    
    class Config:
        from_attributes = True

class ImageResponse(BaseModel):
    """Schema for image response"""
    id: int
    page_num: int
    image_index: int
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[int] = None
    width: int
    height: int
    
    class Config:
        from_attributes = True
