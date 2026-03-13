from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Document(Base):
    """Document metadata table"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)  # User's original name
    file_path = Column(Text, nullable=False)  # Path on disk
    file_size = Column(Integer, nullable=False)  # Size in bytes
    content_type = Column(String(100))  # application/pdf
    
    # Processing status
    status = Column(String(50), default="uploaded")  
    # Status values: uploaded, processing, chunked, indexed, failed
    
    # Metadata
    num_pages = Column(Integer)
    num_chunks = Column(Integer, default=0)

    #Category
    category = Column(String, nullable=True)
    keywords = Column(Text, nullable=True)
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    num_images = Column(Integer, default=0)
    extracted_text_path = Column(Text)  # Path to extracted text file
    has_ocr = Column(Boolean, default=False) # Whether OCR was performed

    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"


class Image(Base):
    """Extracted images from documents"""

    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)

    document_id = Column(
        Integer,
        ForeignKey("documents.id"),
        nullable=False,
        index=True
    )

    # -----------------------------
    # Image location
    # -----------------------------
    page_num = Column(Integer, nullable=False, index=True)

    image_index = Column(
        Integer,
        nullable=False
    )

    image_path = Column(
        Text,
        nullable=False
    )

    # -----------------------------
    # OCR results
    # -----------------------------
    ocr_text = Column(
        Text,
        nullable=True
    )

    ocr_confidence = Column(
        Integer,
        nullable=True
    )

    # -----------------------------
    # Vision model caption
    # -----------------------------
    caption = Column(
        Text,
        nullable=True
    )

    # -----------------------------
    # Future embedding storage
    # -----------------------------
    embedding_vector = Column(
        JSON,
        nullable=True
    )

    # -----------------------------
    # Image metadata
    # -----------------------------
    width = Column(Integer, nullable=True)

    height = Column(Integer, nullable=True)

    format = Column(
        String(10),
        nullable=True
    )

    # Optional: useful for duplicate detection later
    image_hash = Column(
        String(64),
        nullable=True,
        index=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    # -----------------------------
    # Relationships
    # -----------------------------
    document = relationship(
        "Document",
        backref="images"
    )

    def __repr__(self):
        return (
            f"<Image(id={self.id}, "
            f"doc_id={self.document_id}, "
            f"page={self.page_num}, "
            f"index={self.image_index})>"
        )

class Chunk(Base):
    """Text chunks extracted from documents"""

    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)

    document_id = Column(
        Integer,
        ForeignKey("documents.id"),
        nullable=False,
        index=True
    )

    chunk_index = Column(Integer, nullable=False)

    has_embedding = Column(Boolean, default=False)  # Whether embedding is generated
    embedding_model = Column(String(100))  # Model used (e.g., "all-MiniLM-L6-v2")
    faiss_index = Column(Integer)  

    text = Column(Text, nullable=False)

    title = Column(String(500), nullable=True)

    page_num = Column(Integer, nullable=True)

    # metadata enrichment
    num_images = Column(Integer, default=0)

    has_diagram = Column(Boolean, default=False)

    has_table = Column(Boolean, default=False)

    token_count = Column(Integer)

    char_start = Column(Integer)

    char_end = Column(Integer)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    document = relationship(
        "Document",
        back_populates="chunks"
    )

    def __repr__(self):
        return f"<Chunk(id={self.id}, doc={self.document_id}, page={self.page_num})>"

class Query(Base):
    """Query analytics table"""
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    
    # Performance metrics
    latency_ms = Column(Integer)  # Response time in milliseconds
    num_chunks_retrieved = Column(Integer)  # How many chunks were used
    cache_hit = Column(Boolean, default=False)
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<Query(id={self.id}, cache_hit={self.cache_hit}, latency={self.latency_ms}ms)>"
