import httpx
import logging
from typing import AsyncGenerator, Optional, List
import json
import os

from groq import Groq

logger = logging.getLogger(__name__)


class OllamaLLM:
    """
    Unified LLM Service
    - Groq (primary for speed)
    - Ollama (fallback)
    - Summarization
    - Categorization
    """

    def __init__(
        self,
        base_url: str = "http://ollama:11434",
        model: str = "llama3.2:1b",
        timeout: int = 120
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

        self.groq_api_key = os.getenv("GROQ_API_KEY")

        if self.groq_api_key:
            self.groq = Groq(api_key=self.groq_api_key)
            logger.info("⚡ Groq enabled")

        logger.info(f"🤖 Ollama fallback model: {model}")

    # ---------------------------------------------------------
    # Core generation
    # ---------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None
    ) -> str:

        logger.info(f"🔄 Generating response: '{prompt[:50]}...'")

        # ---------- GROQ ----------
        if self.groq_api_key:
            try:
                messages = []

                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                messages.append({"role": "user", "content": prompt})

                response = self.groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                answer = response.choices[0].message.content
                return answer

            except Exception as e:
                logger.warning(f"⚠️ Groq failed, falling back to Ollama: {e}")

        # ---------- OLLAMA FALLBACK ----------
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()

                result = response.json()
                return result.get("response", "")

        except httpx.TimeoutException:
            logger.error("⏱️ LLM request timed out")
            raise Exception("LLM request timed out")

    # ---------------------------------------------------------
    # Streaming
    # ---------------------------------------------------------

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:

                async for line in response.aiter_lines():

                    if line.strip():
                        try:
                            chunk = json.loads(line)

                            if "response" in chunk:
                                yield chunk["response"]

                        except json.JSONDecodeError:
                            continue

    # ---------------------------------------------------------
    # Document summarization
    # ---------------------------------------------------------

    async def summarize_document(
        self,
        chunks: List[str]
    ) -> str:

        logger.info(f"📄 Summarizing {len(chunks)} chunks")

        summaries = []

        for chunk in chunks:

            prompt = f"""
Summarize this document section:

{chunk}

Summary:
"""

            summary = await self.generate(prompt)
            summaries.append(summary)

        combined = "\n".join(summaries)

        final_prompt = f"""
Combine these summaries into a coherent document summary.

{combined}

Final summary:
"""

        final_summary = await self.generate(final_prompt)

        return final_summary

    # ---------------------------------------------------------
    # Categorization
    # ---------------------------------------------------------

    async def categorize_document(self, text: str):

        prompt = f"""
Analyze this document and return JSON with:

- categories
- main_topic
- difficulty
- keywords

Document preview:

{text[:2000]}

Return ONLY valid JSON. Do not include explanations.
"""

        response = await self.generate(prompt)

        try:
            return json.loads(response)

        except Exception:

            logger.warning("⚠️ Failed to parse categorization")

            return {
                "categories": [],
                "main_topic": "unknown",
                "difficulty": "unknown",
                "keywords": []
            }

    # ---------------------------------------------------------
    # Health check
    # ---------------------------------------------------------

    async def health_check(self) -> bool:

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except:
            return False


# ---------------------------------------------------------
# Global Singleton
# ---------------------------------------------------------

_llm_instance = None


def get_llm() -> OllamaLLM:
    global _llm_instance

    if _llm_instance is None:
        _llm_instance = OllamaLLM()

    return _llm_instance