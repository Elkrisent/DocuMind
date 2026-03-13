import fitz  # PyMuPDF
import pytesseract
from PIL import Image as PILImage
from pathlib import Path
import io
from typing import List, Tuple
import logging
from vision.captioner import caption_image
import re
import numpy as np
import os
from vision.groq_captioner import GroqVisionCaptioner

logger = logging.getLogger(__name__)

async def get_image_caption(image_path: str) -> str:
    """Get caption using Groq or BLIP fallback"""

    groq_key = os.getenv("GROQ_API_KEY")
    use_groq = os.getenv("USE_GROQ_VISION", "true").lower() == "true"

    if groq_key and use_groq:
        try:
            captioner = GroqVisionCaptioner(groq_key)
            caption = await captioner.caption_image(image_path)
            logger.info(f"✅ Groq caption generated")
            return caption
        except Exception as e:
            logger.warning(f"⚠️ Groq failed, falling back to BLIP: {e}")

    from vision.captioner import generate_caption
    caption = await generate_caption(image_path)
    return caption


class PDFExtractor:
    """Extract text and images from PDFs with OCR + diagram captioning"""

    def __init__(self, pdf_path: str, output_dir: str = "/documents/images"):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.doc = None
    
    def _is_probable_diagram(self, width: int, height: int) -> bool:
        """
        Heuristic to detect diagrams and skip decorative images
        """

        area = width * height

        # Skip tiny images (icons, bullets)
        if area < 120000:
            return False

        # Skip extreme aspect ratios (lines, bars)
        ratio = width / height

        if ratio < 0.3 or ratio > 4:
            return False

        # Skip long thin shapes
        if width < 250 or height < 250:
            return False

        return True
    
    def _extract_slide_title(self, page_text: str) -> str:

        lines = page_text.split("\n")

        for line in lines:

            line = line.strip()

            if len(line) > 5 and len(line) < 120:
                return line

        return ""

    async def extract_all(self) -> dict:

        try:
            self.doc = fitz.open(self.pdf_path)

            text_content = await self._extract_text()

            images_data = await self._extract_images_with_ocr()

            ocr_text = "\n\n".join(
                [img["ocr_text"] for img in images_data if img["ocr_text"]]
            )

            combined_text = f"{text_content}\n\n--- Text from Images ---\n\n{ocr_text}"

            # ADD DIAGRAM CAPTIONS HERE
            for img in images_data:

                caption = img.get("caption")

                if caption:

                    page = img.get("page_num")

                    page_text = self.doc[page - 1].get_text()

                    title = self._extract_slide_title(page_text)

                    combined_text += (
                        f"\n\nDiagram (Page {page})"
                        f"\nSlide: {title}"
                        f"\nDescription: {caption}"
                    )

            return {
                "text": text_content,
                "num_pages": len(self.doc),
                "images": images_data,
                "combined_text": combined_text,
                "num_images": len(images_data),
            }

        finally:
            if self.doc:
                self.doc.close()

    async def _extract_text(self) -> str:
        """Extract all text from PDF"""

        text_parts = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()

            if text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

        return "\n\n".join(text_parts)

    async def _extract_images_with_ocr(self) -> List[dict]:
        """Extract images from PDF pages and perform OCR + captioning"""

        images_data = []
        seen_xrefs = set()

        for page_num in range(len(self.doc)):

            page = self.doc[page_num]
            page_text = page.get_text()

            image_list = page.get_images(full=True)

            if not image_list:
                continue

            logger.debug(f"Page {page_num + 1}: Found {len(image_list)} images")

            for img_index, img_info in enumerate(image_list):

                xref = img_info[0]

                if xref in seen_xrefs:
                    continue

                seen_xrefs.add(xref)

                try:

                    base_image = self.doc.extract_image(xref)

                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    img = PILImage.open(io.BytesIO(image_bytes))
                    gray = np.array(img.convert("L"))
                    variance = gray.var()

                    # Skip images that are too simple (likely icons)
                    if variance < 400:
                        continue

                    width, height = img.size

                    # Resize very large images to speed up captioning
                    if max(width, height) > 1500:
                        img.thumbnail((1500, 1500))
                        width, height = img.size

                    if not self._is_probable_diagram(width, height):
                        continue

                    image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                    image_path = self.output_dir / image_filename

                    img.save(image_path)

                    # OCR only if page has no selectable text
                    if page_text.strip():
                        ocr_text = ""
                        confidence = 0
                    else:
                        ocr_text, confidence = await self._ocr_image(image_bytes)
                    
                    # NEW: Generate diagram caption
                    caption = ""

                    # Only caption real diagrams and avoid pages with heavy text
                    if self._is_probable_diagram(width, height) and len(page_text) <= 2000:
                        caption = await get_image_caption(str(image_path))

                    images_data.append({
                        "page_num": page_num + 1,
                        "image_index": img_index,
                        "image_path": str(image_path),
                        "ocr_text": ocr_text,
                        "ocr_confidence": confidence,
                        "caption": caption,
                        "width": width,
                        "height": height,
                        "format": image_ext
                    })

                    logger.debug(
                        f"Image {img_index + 1}: OCR {confidence}% | Caption generated"
                    )

                except Exception as e:
                    logger.error(
                        f"Error extracting image {img_index + 1} from page {page_num + 1}: {e}"
                    )
                    continue

        return images_data

    async def _ocr_image(self, image_bytes: bytes) -> Tuple[str, int]:
        """Perform OCR on image"""

        try:

            img = PILImage.open(io.BytesIO(image_bytes))

            if img.mode != "RGB":
                img = img.convert("RGB")

            ocr_data = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT
            )

            text_parts = []
            confidences = []

            for i, word in enumerate(ocr_data["text"]):

                if word.strip():

                    text_parts.append(word)

                    conf = int(ocr_data["conf"][i])
                    if conf > 0:
                        confidences.append(conf)

            text = " ".join(text_parts)

            avg_confidence = (
                int(sum(confidences) / len(confidences))
                if confidences
                else 0
            )

            return text, avg_confidence

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return "", 0


async def process_pdf_document(pdf_path: str, document_id: int) -> dict:
    """
    Main function to process a PDF document
    """

    output_dir = f"/documents/images/doc_{document_id}"

    extractor = PDFExtractor(pdf_path, output_dir)

    results = await extractor.extract_all()

    return results

def clean_text(text: str):

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)

    return text.strip()