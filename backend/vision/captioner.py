from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import logging

logger = logging.getLogger(__name__)

logger.info("Loading BLIP model...")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model.to("cpu")
model.eval()

logger.info("BLIP loaded")


def caption_image(image_path: str) -> str:

    try:
        image = Image.open(image_path).convert("RGB")

        inputs = processor(image, return_tensors="pt")

        with torch.no_grad():
            out = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

        caption = processor.decode(out[0], skip_special_tokens=True)

        return caption

    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        return ""