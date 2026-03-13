import tiktoken
from typing import List, Dict
import re
import logging
from text_processing.cleaner import clean_extracted_text, should_skip_chunk

logger = logging.getLogger(__name__)


class TextChunker:

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    # ------------------------------------------------
    # MAIN ENTRY
    # ------------------------------------------------
    def _is_heading(self, line: str) -> bool:

        line = line.strip()

        if not line:
            return False

        # short lines are often headings
        if len(line) < 80:

            # Title Case
            if line.istitle():
                return True

            # ALL CAPS
            if line.isupper():
                return True

            # ending with colon
            if line.endswith(":"):
                return True

        return False
    

    def _build_slide_chunk(self, title: str, body: str) -> str:
        """
        Combine slide title and body into structured chunk text
        """

        if title:
            return f"Slide Title: {title}\n\n{body}"

        return body

    def chunk_text(self, text: str, document_id=None, mode="auto", images=None):

        if images is None:
            images = []

        if mode == "slides":
            return self.chunk_by_slides(text, document_id, images)

        if mode == "book":
            return self.chunk_by_sections(text, document_id)

        if mode == "auto":

            if self._is_slide_document(text):
                logger.info("Detected slide document")
                return self.chunk_by_slides(text, document_id, images)

            logger.info("Detected book/text document")
            return self.chunk_by_sentences(text, document_id)

    # ------------------------------------------------
    # SLIDE CHUNKING
    # ------------------------------------------------
    def _extract_slide_structure(self, slide_text):
        """
        Extract title and body from a slide page
        """

        lines = slide_text.split("\n")

        title = None
        body_lines = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # first meaningful line becomes title
            if title is None and len(line) < 120:
                title = line
                continue

            body_lines.append(line)

        body = "\n".join(body_lines)

        return title, body

    def chunk_by_slides(self, text: str, document_id: int = None, images=None) -> List[Dict]:
        """
        Chunk text by slide boundaries (optimized for presentations)
        """

        # Better slide split pattern
        slide_pattern = r'(?=---\s*Page\s+\d+\s*---)'
        slides = re.split(slide_pattern, text)
        slides = [s.strip() for s in slides if s.strip()]

        chunks = []
        chunk_index = 0
        char_pos = 0

        for slide in slides:

            page_num = self._extract_page_number(slide)

            title, body = self._extract_slide_structure(slide)

            structured_text = self._build_slide_chunk(title, body)

            # ----------------------------------------
            # Attach diagram captions from images
            # ----------------------------------------

            page_captions = []

            if images:
                for img in images:
                    if img.get("page_num") == page_num:
                        caption = img.get("caption") or img.get("ocr_text")
                        if caption:
                            page_captions.append(caption)

            if page_captions:

                structured_text += "\n\nDiagrams on this slide:\n"

                for cap in page_captions:
                    structured_text += f"- {cap}\n"

            tokens = len(self.encoding.encode(structured_text))

            # ----------------------------------------
            # Large slide fallback
            # ----------------------------------------

            if tokens > self.chunk_size * 1.5:

                slide_chunks = self._chunk_large_text(
                    structured_text,
                    document_id,
                    chunk_index,
                    char_pos,
                    page_num,
                    title
                )

                chunks.extend(slide_chunks)

                chunk_index += len(slide_chunks)
                char_pos += len(structured_text)

                continue

            # ----------------------------------------
            # Normal slide chunk
            # ----------------------------------------

            chunk = {
                "chunk_index": chunk_index,
                "text": structured_text,
                "title": title,
                "token_count": tokens,
                "document_id": document_id,
                "page_num": page_num,

                # metadata enrichment
                "num_images": len(page_captions),
                "has_diagram": len(page_captions) > 0,
                "has_table": self._detect_table(structured_text),

                "char_start": char_pos,
                "char_end": char_pos + len(structured_text)
            }

            chunks.append(chunk)

            chunk_index += 1
            char_pos += len(structured_text)

        logger.info(f"Created {len(chunks)} slide-based chunks")

        return chunks

    # ------------------------------------------------
    # BOOK CHUNKING
    # ------------------------------------------------
    def chunk_by_sentences(self, text, document_id=None):

        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        char_pos = 0

        title = None  # section title

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # Detect headings
            if self._is_heading(sentence):
                title = sentence
                continue

            tokens = len(self.encoding.encode(sentence))

            # If adding this sentence exceeds chunk size
            if current_tokens + tokens > self.chunk_size and current_chunk:

                chunk_text = " ".join(current_chunk)

                chunks.append({
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "title": title,
                    "token_count": current_tokens,
                    "document_id": document_id,
                    "page_num": None,

                    "num_images": 0,
                    "has_diagram": False,
                    "has_table": self._detect_table(chunk_text),

                    "char_start": char_pos - len(chunk_text),
                    "char_end": char_pos
                })

                overlap = self._get_overlap_sentences(
                    current_chunk,
                    self.chunk_overlap
                )

                current_chunk = overlap
                current_tokens = sum(
                    len(self.encoding.encode(s)) for s in overlap
                )

                chunk_index += 1

            current_chunk.append(sentence)
            current_tokens += tokens
            char_pos += len(sentence) + 1

        # Final chunk
        if current_chunk:

            chunk_text = " ".join(current_chunk)

            chunks.append({
                "chunk_index": chunk_index,
                "text": chunk_text,
                "title": title,
                "token_count": current_tokens,
                "document_id": document_id,
                "page_num": None,

                "num_images": 0,
                "has_diagram": False,
                "has_table": self._detect_table(chunk_text),

                "char_start": char_pos - len(chunk_text),
                "char_end": char_pos
            })

        logger.info(f"Created {len(chunks)} book chunks")

        return chunks
    # ------------------------------------------------
    # LARGE TEXT FALLBACK
    # ------------------------------------------------
    def _chunk_large_text(
        self,
        text,
        document_id,
        chunk_index_start,
        char_pos,
        page_num,
        title
    ):

        sentences = self._split_into_sentences(text)

        chunks = []
        current = []
        tokens = 0
        chunk_index = chunk_index_start

        for sentence in sentences:

            s_tokens = len(self.encoding.encode(sentence))

            if tokens + s_tokens > self.chunk_size and current:

                chunk_text = " ".join(current)

                chunks.append({
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "title": title,
                    "token_count": tokens,
                    "document_id": document_id,
                    "page_num": page_num,

                    "num_images": 0,
                    "has_diagram": False,
                    "has_table": self._detect_table(chunk_text),

                    "char_start": char_pos,
                    "char_end": char_pos + len(chunk_text)
                })

                current = []
                tokens = 0
                chunk_index += 1

            current.append(sentence)
            tokens += s_tokens

        if current:

            chunk_text = " ".join(current)

            chunks.append({
                "chunk_index": chunk_index,
                "text": chunk_text,
                "title": title,
                "token_count": tokens,
                "document_id": document_id,
                "page_num": page_num,

                "num_images": 0,
                "has_diagram": False,
                "has_table": self._detect_table(chunk_text),

                "char_start": char_pos,
                "char_end": char_pos + len(chunk_text)
            })

        return chunks

    # ------------------------------------------------
    # UTILITIES
    # ------------------------------------------------
    def _split_into_sentences(self, text):

        text = text.replace("Dr.", "Dr<DOT>")
        text = text.replace("Mr.", "Mr<DOT>")
        text = text.replace("Mrs.", "Mrs<DOT>")

        sentences = re.split(r'[.!?]+\s+', text)

        sentences = [
            s.replace("<DOT>", ".").strip()
            for s in sentences
            if s.strip()
        ]

        return sentences

    def _get_overlap_sentences(self, sentences, max_tokens):

        overlap = []
        tokens = 0

        for sentence in reversed(sentences):

            s_tokens = len(self.encoding.encode(sentence))

            if tokens + s_tokens > max_tokens:
                break

            overlap.insert(0, sentence)
            tokens += s_tokens

        return overlap

    def _extract_page_number(self, text):

        match = re.search(r'--- Page (\d+) ---', text)

        if match:
            return int(match.group(1))

        return None

    def _detect_table(self, text):

        if "|" in text:
            return True

        if re.search(r'\w+\s{2,}\w+', text):
            return True

        if re.search(r'\d+\s+\d+', text):
            return True

        return False

    def _is_slide_document(self, text):

        pages = text.count("--- Page ")

        if pages == 0:
            return False

        avg_chars = len(text) / pages

        return avg_chars < 1200


def chunk_document_text(text: str, document_id: int = None, images=None) -> List[Dict]:
    """Main function to chunk document text"""

    # Clean text first
    text = clean_extracted_text(text)

    chunker = TextChunker(chunk_size=800, chunk_overlap=200)

    # Check if this looks like slides
    if "Slide Title:" in text:
        logger.info("📊 Detected slide format, using slide-aware chunking")
        chunks = chunker.chunk_by_slides(
            text,
            document_id=document_id,
            images=images
        )
    else:
        logger.info("📄 Using standard semantic chunking")
        chunks = chunker.chunk_text(text, document_id)

    # Filter low-quality chunks
    quality_chunks = [
        chunk for chunk in chunks
        if not should_skip_chunk(chunk["text"])
    ]

    return quality_chunks