import re


# ------------------------------------------------
# PROTECT SPECIAL CONTENT
# ------------------------------------------------

def protect_special_content(text: str):
    """
    Protect math, grammar rules, and code-like blocks
    so preprocessing does not modify them.
    """

    protected_blocks = []

    def replacer(match):
        protected_blocks.append(match.group(0))
        return f"__PROTECTED_{len(protected_blocks)-1}__"

    # Grammar rules / rewrite rules
    text = re.sub(r'[^\n]*→[^\n]*', replacer, text)

    # Equations
    text = re.sub(r'\b[A-Za-z0-9]+\s*=\s*[^\n]+', replacer, text)

    # Code blocks inside braces
    text = re.sub(r'\{[^}]*\}', replacer, text)

    return text, protected_blocks


def restore_protected_content(text: str, protected_blocks):
    """
    Restore protected content.
    """

    for i, block in enumerate(protected_blocks):
        text = text.replace(f"__PROTECTED_{i}__", block)

    return text


# ------------------------------------------------
# CORE TEXT CLEANING
# ------------------------------------------------

def clean_extracted_text(text: str) -> str:
    """
    Clean raw PDF extraction artifacts while preserving meaning.
    """

    if not text:
        return ""

    # Remove page markers
    text = re.sub(r'-{2,}\s*Page\s+\d+\s*-{2,}', '', text, flags=re.IGNORECASE)

    # Remove isolated page numbers
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)

    # Remove emails / footer lines
    text = re.sub(r'(?m)^\S+@\S+\.\S+\s*$', '', text)

    # Fix hyphenated line breaks
    text = re.sub(r'-\n', '', text)

    # Join wrapped lines that break mid-sentence
    text = re.sub(r'(?<![\.\:\n])\n(?!\n)', ' ', text)

    # Normalize bullet characters
    text = re.sub(r'[•◦▪●○■]', '-', text)

    # Normalize "o bullet"
    text = re.sub(r'(?m)^\s*o\s+', '- ', text)

    # Remove stray bullet-only lines
    text = re.sub(r'(?m)^\s*-\s*$', '', text)

    # Remove strange unicode replacement chars
    text = text.replace("�", "")

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)

    # Normalize newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Clean paragraph spacing
    text = re.sub(r'\n\s*\n', '\n\n', text)

    text = re.sub(r'\n-\s*\n', '\n- ', text)

    text = re.sub(r' -\s*\n', '\n- ', text)

    text = re.sub(r'(?<![.\n])\n(?!\n|-)', ' ', text)

    return text.strip()


# ------------------------------------------------
# TABLE FORMATTING
# ------------------------------------------------

def format_tables(text: str) -> str:
    """
    Convert simple column tables into structured key-value lines.
    """

    lines = text.split("\n")
    formatted = []

    for line in lines:

        # Split by large spacing (common PDF table layout)
        columns = re.split(r'\s{2,}', line.strip())

        if len(columns) >= 2:

            pairs = []

            for i in range(len(columns) - 1):

                key = columns[i].strip()
                value = columns[i + 1].strip()

                if key and value:
                    pairs.append(f"{key}: {value}")

            if pairs:
                formatted.append(" | ".join(pairs))
                continue

        formatted.append(line)

    return "\n".join(formatted)


# ------------------------------------------------
# CHUNK QUALITY FILTER
# ------------------------------------------------

def should_skip_chunk(text: str, min_words: int = 20) -> bool:
    """
    Determine if a chunk is low quality and should be skipped.
    """

    if not text:
        return True

    words = text.split()

    # Too short
    if len(words) < min_words:
        return True

    # Mostly symbols
    alpha_chars = sum(c.isalpha() for c in text)

    if alpha_chars < len(text) * 0.3:
        return True

    # Common slide noise
    noise_patterns = [
        r'^THANK YOU\s*$',
        r'^Questions\?\s*$',
        r'^References\s*$',
        r'^Conclusion\s*$',
        r'^\d+\s*$'
    ]

    for pattern in noise_patterns:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True

    return False


# ------------------------------------------------
# FULL PREPROCESSING PIPELINE
# ------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for ingestion.
    """

    # Protect math/code
    text, protected = protect_special_content(text)

    # Clean PDF artifacts
    text = clean_extracted_text(text)

    # Format tables
    text = format_tables(text)

    # Restore protected content
    text = restore_protected_content(text, protected)

    return text