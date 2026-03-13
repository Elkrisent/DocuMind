import httpx
import base64
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class GroqVisionCaptioner:
    """
    Fast vision captioning using Groq API (free tier)
    Much better than BLIP, 10x faster on CPU
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    async def caption_image(self, image_path: str) -> str:
        """Generate caption for image"""
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Build request
        payload = {
            "model": "llama-3.2-90b-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this diagram/image in one concise sentence. Focus on what it shows, not aesthetic details."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 100,
            "temperature": 0.3
        }
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                result = response.json()
                caption = result['choices'][0]['message']['content']
                
                logger.info(f"✅ Generated caption: {caption[:50]}...")
                return caption
                
        except Exception as e:
            logger.error(f"Groq vision error: {e}")
            # Fallback to BLIP
            return await self._fallback_blip(image_path)
    
    async def _fallback_blip(self, image_path: str) -> str:
        """Fallback to BLIP if API fails"""
        from vision.captioner import generate_caption
        return await generate_caption(image_path)