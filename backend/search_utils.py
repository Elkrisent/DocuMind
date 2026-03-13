import re

def keyword_score(query: str, text: str) -> float:
    """
    Simple keyword overlap scoring
    """

    q_words = query.lower().split()
    text = text.lower()

    matches = sum(1 for w in q_words if w in text)

    return matches / max(len(q_words), 1)

def clean_result_text(text: str) -> str:
    """Clean chunk text before returning in API"""

    # Remove page markers
    text = re.sub(r'---\s*Page\s+\d+\s*---', '', text)

    # Remove weird characters
    text = re.sub(r'[�]', '', text)

    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)

    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def looks_like_definition(text: str) -> bool:
    """
    Detect if a chunk looks like a definition.
    Works for textbooks and lecture slides.
    """

    text = text.lower()

    patterns = [
        r"\bis\b",
        r"\brefers to\b",
        r"\bmeans\b",
        r"\bis defined as\b",
        r"\bcan be defined as\b"
    ]

    for p in patterns:
        if re.search(p, text):
            return True

    return False