import re
import tiktoken
from typing import List, Dict, Literal

class AdaptiveChunker:
    """
    Context-aware chunking for slides, textbooks, and PDFs
    """
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        doc_type: Literal["slides", "textbook", "paper", "auto"] = "auto"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.doc_type = doc_type
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk_document(self, text: str, images: List[dict] = None) -> List[Dict]:
        """Auto-detect document type and chunk accordingly"""
        
        # Auto-detect document type
        if self.doc_type == "auto":
            doc_type = self._detect_document_type(text)
        else:
            doc_type = self.doc_type
        
        # Route to appropriate chunker
        if doc_type == "slides":
            return self._chunk_slides(text, images)
        elif doc_type == "textbook":
            return self._chunk_textbook(text)
        else:
            return self._chunk_paper(text)
    
    def _detect_document_type(self, text: str) -> str:
        """Detect if document is slides, textbook, or paper"""
        
        # Slide indicators
        slide_markers = text.count("Slide Title:")
        page_markers = text.count("--- Page")
        
        if slide_markers > 5:
            return "slides"
        
        # Textbook indicators (chapters, sections, exercises)
        chapter_count = len(re.findall(r'\bChapter \d+', text, re.IGNORECASE))
        exercise_count = len(re.findall(r'\bExercise \d+', text, re.IGNORECASE))
        
        if chapter_count > 2 or exercise_count > 5:
            return "textbook"
        
        return "paper"
    
    def _chunk_slides(self, text: str, images: List[dict] = None) -> List[Dict]:
        """
        Slide-aware chunking - one or more slides per chunk
        Keeps slides intact, combines small ones
        """
        
        # Split by slide markers
        slide_pattern = r'Slide Title:\s*---\s*Page\s+\d+\s*---?\s*\n'
        slides = re.split(slide_pattern, text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for i, slide in enumerate(slides):
            if not slide.strip():
                continue
            
            slide_tokens = len(self.encoding.encode(slide))
            
            # If single slide exceeds max size, split it
            if slide_tokens > self.chunk_size * 1.5:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, chunk_index))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Split large slide by paragraphs
                sub_chunks = self._split_large_slide(slide, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            
            # If adding this slide exceeds size, save current chunk
            elif current_tokens + slide_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, chunk_index))
                    chunk_index += 1
                
                # Start new chunk with this slide
                current_chunk = [slide]
                current_tokens = slide_tokens
            
            # Add slide to current chunk
            else:
                current_chunk.append(slide)
                current_tokens += slide_tokens
        
        # Don't forget last chunk
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, chunk_index))
        
        return chunks
    
    def _chunk_textbook(self, text: str) -> List[Dict]:
        """
        Textbook chunking - respect sections and paragraphs
        """
        
        # First try to split by sections
        section_pattern = r'\n##?\s+\d+\.?\d*\s+[A-Z]'  # "## 2.1 Introduction" etc
        sections = re.split(section_pattern, text)
        
        chunks = []
        chunk_index = 0
        
        for section in sections:
            if not section.strip():
                continue
            
            section_tokens = len(self.encoding.encode(section))
            
            # If section fits in one chunk, keep it
            if section_tokens <= self.chunk_size:
                chunks.append({
                    'chunk_index': chunk_index,
                    'text': section.strip(),
                    'token_count': section_tokens,
                    'type': 'section'
                })
                chunk_index += 1
            
            # Otherwise split by paragraphs with overlap
            else:
                para_chunks = self._split_by_paragraphs(section, chunk_index)
                chunks.extend(para_chunks)
                chunk_index += len(para_chunks)
        
        return chunks
    
    def _split_by_paragraphs(self, text: str, start_index: int) -> List[Dict]:
        """Split text by paragraphs with overlap"""
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = start_index
        
        for para in paragraphs:
            para_tokens = len(self.encoding.encode(para))
            
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'chunk_index': chunk_index,
                    'text': chunk_text,
                    'token_count': current_tokens,
                    'type': 'paragraph_group'
                })
                chunk_index += 1
                
                # Keep last paragraph for overlap
                if self.chunk_overlap > 0:
                    overlap_para = current_chunk[-1]
                    current_chunk = [overlap_para, para]
                    current_tokens = len(self.encoding.encode(overlap_para)) + para_tokens
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'chunk_index': chunk_index,
                'text': chunk_text,
                'token_count': current_tokens,
                'type': 'paragraph_group'
            })
        
        return chunks
    
    def _split_large_slide(self, slide: str, start_index: int) -> List[Dict]:
        """Split a large slide by bullet points or paragraphs"""
        
        # Try splitting by bullet points first
        if '•' in slide or '*' in slide or '-' in slide[:100]:
            parts = re.split(r'\n[•\*\-]\s+', slide)
        else:
            parts = slide.split('\n\n')
        
        return self._split_by_paragraphs('\n\n'.join(parts), start_index)
    
    def _create_chunk(self, content_list: List[str], index: int) -> Dict:
        """Helper to create chunk dict"""
        text = '\n\n'.join(content_list)
        return {
            'chunk_index': index,
            'text': text.strip(),
            'token_count': len(self.encoding.encode(text)),
            'type': 'slide_group'
        }
    
    def _chunk_paper(self, text: str) -> List[Dict]:
        """Academic paper chunking - respect abstract, sections"""
        # Similar to textbook but look for abstract, introduction, etc.
        return self._chunk_textbook(text)  # Simplified for now