from sentence_transformers import CrossEncoder

_reranker = None


def get_reranker():

    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(
            "BAAI/bge-reranker-base",
            max_length=512
        )

    return _reranker


def rerank_results(query, results):

    reranker = get_reranker()

    pairs = [
        (query, r["text"])
        for r in results
    ]

    scores = reranker.predict(pairs)

    for r, score in zip(results, scores):
        r["rerank_score"] = float(score)

    results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results