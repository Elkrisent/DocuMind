import re


def normalize_query(query: str) -> str:
    """
    Normalize user queries for better retrieval.
    Keeps the query general so it works across domains.
    """

    query = query.lower().strip()

    # remove punctuation
    query = re.sub(r"[^\w\s]", " ", query)

    # collapse whitespace
    query = re.sub(r"\s+", " ", query)

    return query


def expand_query(query: str) -> str:
    """
    Light query expansion for common question patterns.
    """

    q = normalize_query(query)

    expansions = []

    # remove common question prefixes
    prefixes = [
        "what is",
        "what are",
        "what does",
        "what do",
        "define",
        "explain",
        "describe"
    ]

    for p in prefixes:
        if q.startswith(p):
            expansions.append(q.replace(p, "").strip())

    # difference queries
    if "difference between" in q:
        parts = q.replace("difference between", "").split("and")
        if len(parts) == 2:
            a = parts[0].strip()
            b = parts[1].strip()
            expansions.append(f"{a} {b} comparison")

    expanded = " ".join([q] + expansions)

    return expanded.strip()