import math

def evaluate(results, qrels, k=10):
    """
    results: {query_id: [doc_id, doc_id, ...]}   best result first
    qrels:   {query_id: {doc_id: relevance}}
    """
    recall, mrr, ndcg, n = 0.0, 0.0, 0.0, 0

    for qid, ranked in results.items():
        relevant = {d for d, r in qrels.get(qid, {}).items() if r > 0}
        if not relevant:
            continue
        n += 1
        top = ranked[:k]

        # Recall@k: fraction of relevant docs found in the top k
        recall += len(relevant & set(top)) / len(relevant)

        # MRR@k: 1 / rank of the first relevant doc
        for i, d in enumerate(top):
            if d in relevant:
                mrr += 1 / (i + 1)
                break

        # nDCG@k: reward relevant docs near the top
        dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(top) if d in relevant)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        ndcg += dcg / ideal

    return {
        f"Recall@{k}": round(recall / n, 4),
        f"MRR@{k}": round(mrr / n, 4),
        f"nDCG@{k}": round(ndcg / n, 4),
    }