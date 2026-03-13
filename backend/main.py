from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse  # ← Add this line
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
# ... rest of imports
from sqlalchemy import select, func
from datetime import datetime
import os
import uuid
from retrieval import HybridRetriever
import aiofiles
from pathlib import Path
from sqlalchemy.orm import selectinload 
from extraction import process_pdf_document
import logging
from chunking import chunk_document_text
from database import engine, get_db
from models import Document, Chunk, Query, Base, Image
from text_processing.cleaner import preprocess_text, should_skip_chunk
from embeddings import get_embedding_generator, generate_chunk_embeddings
from vector_store import get_vector_store, save_vector_store
import numpy as np 
from search_utils import clean_result_text, keyword_score
from llm.generator import *
from llm.prompts import build_rag_prompt, build_simple_prompt, RAG_SYSTEM_PROMPT
from chunkingv2 import AdaptiveChunker

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Silence noisy libraries
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# Import our modules
from schemas import (
    DocumentResponse, 
    DocumentDetailResponse, 
    ChunkResponse,
    QueryCreate,
    QueryResponse
)

app = FastAPI(
    title="DocuMind API",
    version="0.2.0",
    description="Intelligent Document Processing System"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File storage configuration
UPLOAD_DIR = Path("/documents")
UPLOAD_DIR.mkdir(exist_ok=True)

# Startup event
@app.on_event("startup")
async def startup():
    """Initialize database on startup"""
    print("🚀 Starting DocuMind API...")
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    await init_db()

@app.get("/")
async def root():
    return {
        "message": "DocuMind API v0.2.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    """Health check with database status"""
    try:
        # Test database connection
        result = await db.execute(select(func.count(Document.id)))
        doc_count = result.scalar()
        
        return {
            "status": "healthy",
            "database": "connected",
            "total_documents": doc_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a PDF document and extract text + images with OCR
    """
    # Validate file type
    if not file.content_type == "application/pdf":
        raise HTTPException(400, "Only PDF files are supported")
    
    # Generate unique filename
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save file to disk
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        file_size = len(content)
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {str(e)}")
    
    # Create database record
    document = Document(
        filename=unique_filename,
        original_filename=file.filename,
        file_path=str(file_path),
        file_size=file_size,
        content_type=file.content_type,
        status="processing"  # Changed from "uploaded"
    )
    
    db.add(document)
    await db.commit()
    await db.refresh(document)
    
    # Process PDF in background (for now, we'll do it synchronously)
    # In Day 4 we'll make this truly async with workers
    try:
        await process_document_extraction(document.id, str(file_path), db)
    except Exception as e:
        logger.error(f"Error processing document {document.id}: {e}")
        document.status = "failed"
        await db.commit()
        raise HTTPException(500, f"Processing failed: {str(e)}")
    
    await db.refresh(document)
    return document

async def process_document_extraction(doc_id: int, pdf_path: str, db: AsyncSession):
    """Process PDF: extract text, images, OCR, captions, chunk, and embed"""

    logger.info(f"📄 Processing Document #{doc_id}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # STEP 1: Extract everything
    logger.info("🔍 Extracting text and images...")
    results = await process_pdf_document(pdf_path, doc_id)

    logger.info(
        f"✅ Extracted: {results['num_pages']} pages | {results['num_images']} images"
    )

    # STEP 2: Fetch document from DB
    result = await db.execute(select(Document).where(Document.id == doc_id))
    document = result.scalar_one()

    # STEP 3: Save extracted text
    text_filename = f"{doc_id}_extracted.txt"
    text_path = UPLOAD_DIR / text_filename

    logger.info("💾 Saving extracted text...")

    async with aiofiles.open(text_path, "w", encoding="utf-8") as f:
        await f.write(results["combined_text"])

    # STEP 4: Update document metadata
    document.num_pages = results["num_pages"]
    document.num_images = results["num_images"]
    document.extracted_text_path = str(text_path)
    document.has_ocr = results["num_images"] > 0
    document.status = "chunking"

    await db.commit()

    # STEP 4.5: Auto-categorize document
    logger.info("🏷️ Categorizing document...")

    llm = get_llm()

    try:
        categories = await llm.categorize_document(results["combined_text"])

        document.category = categories.get("main_topic", "unknown")

        document.keywords = json.dumps(
            categories.get("keywords", [])
        )

        logger.info(
            f"📚 Topic: {categories.get('main_topic')} | "
            f"Categories: {categories.get('categories')}"
        )

    except Exception as e:
        logger.warning(f"Categorization failed: {e}")

    # STEP 5: Save images + OCR results
    if results["images"]:
        logger.info(f"🖼️ Processing {len(results['images'])} images with OCR...")

        for img_data in results["images"]:
            image = Image(
                document_id=doc_id,
                page_num=img_data["page_num"],
                image_index=img_data["image_index"],
                image_path=img_data["image_path"],
                ocr_text=img_data["ocr_text"],
                ocr_confidence=img_data["ocr_confidence"],
                width=img_data["width"],
                height=img_data["height"],
                format=img_data["format"],
            )
            db.add(image)

        avg_conf = (
            sum(i["ocr_confidence"] for i in results["images"]) // len(results["images"])
        )

        logger.info(f"✅ OCR complete (avg confidence: {avg_conf}%)")

    await db.commit()

    # STEP 6: Chunk the text
    logger.info("✂️ Chunking text...")

    clean_text = preprocess_text(results["combined_text"])

    chunker = AdaptiveChunker(
        chunk_size=800,
        chunk_overlap=200,
        doc_type="auto"  # Auto-detect slides vs textbook vs paper
    )
    
    chunks_data = chunker.chunk_document(
        text=clean_text,
        images=results["images"]
    )
    
    # Add document_id to chunks
    for i, chunk in enumerate(chunks_data):
        chunk['document_id'] = doc_id
        if 'chunk_index' not in chunk:
            chunk['chunk_index'] = i
    
    logger.info(f"✅ Created {len(chunks_data)} chunks")

    # STEP 7: Generate embeddings for chunks
    logger.info(f"🧠 Generating embeddings for {len(chunks_data)} chunks...")

    embedding_gen = get_embedding_generator()
    chunk_embeddings = await generate_chunk_embeddings(chunks_data)

    logger.info(f"✅ Generated {len(chunk_embeddings)} embeddings")

    # STEP 8: Save chunks and index in FAISS
    logger.info("📊 Indexing chunks in FAISS...")

    vector_store = get_vector_store()

    chunk_ids = []
    for chunk_data in chunks_data:
        chunk = Chunk(
            document_id=doc_id,
            chunk_index=chunk_data["chunk_index"],
            text=chunk_data["text"],
        )  

        db.add(chunk)
        await db.flush()  # obtain chunk ID
        chunk_ids.append(chunk.id)

    vector_store.add_embeddings(
        chunk_embeddings,
        chunk_ids,
        data_type="chunk"
    )

    save_vector_store()

    logger.info(f"✅ Indexed {len(chunk_ids)} chunks")

    # STEP 9: Generate embeddings for image captions
    if results.get("images"):
        logger.info(f"🖼️ Generating embeddings for {len(results['images'])} captions...")

        image_result = await db.execute(
            select(Image)
            .where(Image.document_id == doc_id)
            .order_by(Image.id)
        )

        images = image_result.scalars().all()

        if images:
            captions = [
                img.caption or img.ocr_text or ""
                for img in images
            ]

            from embeddings import generate_caption_embeddings
            caption_embeddings = await generate_caption_embeddings(captions)

            image_ids = [img.id for img in images]

            vector_store.add_embeddings(
                caption_embeddings,
                image_ids,
                data_type="caption"
            )

            save_vector_store()

            logger.info(f"✅ Indexed {len(image_ids)} captions")

    # STEP 10: Finalize document
    document.num_chunks = len(chunks_data)
    document.status = "indexed"

    await db.commit()

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"🎉 Document #{doc_id} complete!")
    logger.info(
        f"📊 Pages: {results['num_pages']} | Images: {results['num_images']} | Chunks: {len(chunks_data)}"
    )
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

@app.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all documents with pagination"""
    result = await db.execute(
        select(Document)
        .order_by(Document.uploaded_at.desc())
        .offset(skip)
        .limit(limit)
    )
    documents = result.scalars().all()
    return documents

@app.get("/documents/{doc_id}", response_model=DocumentDetailResponse)
async def get_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get specific document with chunks"""
    result = await db.execute(
        select(Document)
        .options(selectinload(Document.chunks))  # ← Add this line
        .where(Document.id == doc_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(404, "Document not found")
    
    return document

@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a document and its file"""
    result = await db.execute(
        select(Document).where(Document.id == doc_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(404, "Document not found")
    
    # Delete file from disk
    try:
        file_path = Path(document.file_path)
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Warning: Could not delete file: {e}")
    
    # Delete from database (cascades to chunks)
    await db.delete(document)
    await db.commit()
    
    return {"message": "Document deleted successfully", "id": doc_id}

@app.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get system statistics"""
    # Count documents
    doc_result = await db.execute(select(func.count(Document.id)))
    total_docs = doc_result.scalar()
    
    # Count chunks
    chunk_result = await db.execute(select(func.count(Chunk.id)))
    total_chunks = chunk_result.scalar()
    
    # Count queries
    query_result = await db.execute(select(func.count(Query.id)))
    total_queries = query_result.scalar()
    
    # Total storage
    size_result = await db.execute(select(func.sum(Document.file_size)))
    total_size = size_result.scalar() or 0
    
    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "total_queries": total_queries,
        "total_storage_bytes": total_size,
        "total_storage_mb": round(total_size / (1024 * 1024), 2)
    }

@app.get("/documents/{doc_id}/chunks", response_model=list[ChunkResponse])
async def get_document_chunks(
    doc_id: int,
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """Get chunks for a specific document"""
    result = await db.execute(
        select(Chunk)
        .where(Chunk.document_id == doc_id)
        .order_by(Chunk.chunk_index)
        .offset(skip)
        .limit(limit)
    )
    chunks = result.scalars().all()
    return chunks

@app.get("/chunks/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(
    chunk_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get specific chunk by ID"""
    result = await db.execute(
        select(Chunk).where(Chunk.id == chunk_id)
    )
    chunk = result.scalar_one_or_none()
    
    if not chunk:
        raise HTTPException(404, "Chunk not found")
    
    return chunk

@app.get("/documents/{doc_id}/text")
async def get_document_text(
    doc_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get full extracted text for a document"""
    result = await db.execute(
        select(Document).where(Document.id == doc_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(404, "Document not found")
    
    if not document.extracted_text_path:
        raise HTTPException(404, "Text not yet extracted")
    
    # Read text file
    try:
        async with aiofiles.open(document.extracted_text_path, 'r', encoding='utf-8') as f:
            text = await f.read()
        return {"text": text, "num_chunks": document.num_chunks}
    except Exception as e:
        raise HTTPException(500, f"Error reading text: {str(e)}")


@app.post("/search")
async def semantic_search(
    query: str,
    k: int = 5,
    use_hybrid: bool = True,  # NEW: Enable hybrid search
    boost_titles: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """
    Hybrid search: BM25 + Semantic + Re-ranking
    """
    
    if not query or not query.strip():
        raise HTTPException(400, "Query cannot be empty")

    logger.info(f"🔍 Searching: '{query}' (hybrid={use_hybrid})")

    # Generate query embedding
    embedding_gen = get_embedding_generator()
    query_embedding = embedding_gen.embed_query(query)

    # Get semantic candidates (more than needed)
    vector_store = get_vector_store()
    semantic_results = vector_store.search(query_embedding, k=k * 4)

    if not semantic_results:
        return {"query": query, "results": [], "total": 0}

    # NEW: Hybrid search with BM25
    if use_hybrid:
        # Get all chunks for BM25 indexing
        chunks_result = await db.execute(
            select(Chunk).options(selectinload(Chunk.document))
        )
        all_chunks = chunks_result.scalars().all()
        
        # Build BM25 index
        retriever = HybridRetriever(alpha=0.6)  # 60% semantic, 40% BM25
        retriever.index_chunks([{
            'id': c.id,
            'text': c.text
        } for c in all_chunks])
        
        # Combine scores
        semantic_chunk_results = [
            (id_val, score) 
            for data_type, id_val, score in semantic_results 
            if data_type == "chunk"
        ]
        
        hybrid_scores = retriever.hybrid_search(
            query,
            semantic_chunk_results,
            k=k * 2
        )
        
        # Convert back to original format
        hybrid_results = [
            ("chunk", chunk_id, score)
            for chunk_id, score in hybrid_scores
        ]
        
        # Add caption results back
        caption_results = [
            r for r in semantic_results if r[0] == "caption"
        ]
        
        results_to_process = hybrid_results + caption_results
    else:
        results_to_process = semantic_results

    # Rest of code continues as before...
    search_results = []
    
    for data_type, id_val, score in results_to_process[:k * 2]:
        if data_type == "chunk":
            result = await db.execute(
                select(Chunk, Document)
                .join(Document)
                .where(Chunk.id == id_val)
            )
            
            row = result.first()
            if not row:
                continue
            
            chunk, document = row
            
            kw_score = keyword_score(query, chunk.text)
            combined_score = 0.7 * score + 0.3 * kw_score
            
            if boost_titles and query.lower() in chunk.text.lower():
                combined_score *= 1.2
            
            clean_text = clean_result_text(chunk.text)
            
            search_results.append({
                "type": "chunk",
                "score": round(combined_score, 4),
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "text": clean_text[:500] + "..." if len(clean_text) > 500 else clean_text,
                "document_id": document.id,
                "document_name": document.original_filename
            })
        
        elif data_type == "caption":
            # ... existing caption handling ...
            pass

    # Filter and sort
    MIN_SCORE = 0.15  # Lowered threshold
    search_results = [r for r in search_results if r["score"] > MIN_SCORE]
    search_results.sort(key=lambda x: x["score"], reverse=True)
    search_results = search_results[:k]
    
    logger.info(f"✅ Found {len(search_results)} results")
    
    return {
        "query": query,
        "results": search_results,
        "total": len(search_results)
    }


@app.get("/index/stats")
async def get_index_stats():
    """Get vector index statistics"""
    vector_store = get_vector_store()
    stats = vector_store.get_stats()
    return stats

@app.post("/ask")
async def ask_question(
    query: str,
    k: int = 5,
    stream: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    RAG-powered question answering
    
    Args:
        query: User question
        k: Number of context chunks to retrieve
        stream: Whether to stream the response
        
    Returns:
        Answer with sources or streaming response
    """
    
    if not query or not query.strip():
        raise HTTPException(400, "Query cannot be empty")
    
    logger.info(f"💭 Question: '{query}'")
    
    # STEP 1: Retrieve relevant context
    logger.info(f"🔍 Retrieving {k} relevant chunks...")
    search_results = await semantic_search(query, k=k, db=db)
    
    if not search_results["results"]:
        return {
            "answer": "I don't have any relevant information to answer that question. Please upload relevant documents first.",
            "sources": [],
            "context_used": False
        }
    
    # STEP 2: Assemble context from chunks
    logger.info("📝 Assembling context...")
    context_parts = []
    sources = []
    
    for i, result in enumerate(search_results["results"], 1):
        if result["type"] == "chunk":
            context_parts.append(
                f"[{i}] From {result['document_name']}:\n{result['text']}\n"
            )
            sources.append({
                "id": i,
                "document": result["document_name"],
                "type": "text chunk",
                "score": result["score"]
            })
        elif result["type"] == "caption":
            context_parts.append(
                f"[{i}] Image from {result['document_name']} (page {result['page_num']}):\n{result['caption']}\n"
            )
            sources.append({
                "id": i,
                "document": result["document_name"],
                "type": "image caption",
                "page": result["page_num"],
                "score": result["score"]
            })
    
    context = "\n\n".join(context_parts)
    
    # STEP 3: Build prompt
    prompt = build_rag_prompt(query, context)
    
    # STEP 4: Generate answer
    llm = get_llm()
    
    if stream:
        # Streaming response
        async def generate():
            try:
                async for token in llm.stream(
                    prompt,
                    system_prompt=RAG_SYSTEM_PROMPT,
                    temperature=0.7
                ):
                    yield token
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"\n\n[Error: {str(e)}]"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain"
        )
    else:
        # Complete response
        try:
            logger.info("🤖 Generating answer...")
            answer = await llm.generate(
                prompt,
                system_prompt=RAG_SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=1024
            )
            
            logger.info("✅ Answer generated")
            
            return {
                "answer": answer.strip(),
                "sources": sources,
                "context_used": True,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(500, f"Failed to generate answer: {str(e)}")


@app.post("/ask/simple")
async def ask_simple(query: str):
    """
    Simple LLM query without RAG (for testing)
    """
    
    if not query or not query.strip():
        raise HTTPException(400, "Query cannot be empty")
    
    llm = get_llm()
    prompt = build_simple_prompt(query)
    
    try:
        answer = await llm.generate(prompt, temperature=0.7)
        return {"answer": answer.strip()}
    except Exception as e:
        raise HTTPException(500, f"LLM error: {str(e)}")


@app.get("/llm/health")
async def llm_health():
    """Check if LLM is available"""
    llm = get_llm()
    is_healthy = await llm.health_check()
    
    return {
        "status": "healthy" if is_healthy else "unavailable",
        "model": llm.model,
        "base_url": llm.base_url
    }

@app.post("/documents/{doc_id}/summarize")
async def summarize_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a summary of a document using its chunks
    """

    # Fetch chunks
    result = await db.execute(
        select(Chunk)
        .where(Chunk.document_id == doc_id)
        .order_by(Chunk.chunk_index)
    )

    chunks = result.scalars().all()

    if not chunks:
        raise HTTPException(404, "Document has no chunks")

    # Build chunk list
    chunk_texts = [c.text for c in chunks]

    llm = get_llm()

    try:
        summary = await llm.summarize_document(chunk_texts)

        return {
            "document_id": doc_id,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(500, f"Summarization failed: {str(e)}")

@app.get("/llm/test-groq")
async def test_groq():

    llm = get_llm()

    try:
        response = await llm.generate(
            "Explain transformers in one sentence."
        )

        return {
            "status": "success",
            "response": response
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)