import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, doc_ids, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.doc_ids = doc_ids
        self.model = SentenceTransformer(model_name)
        self.doc_emb = None                     # filled by build() or load()

    def build(self, texts, batch_size=64):
        """Embed all documents. Vectors are normalized so dot product = cosine."""
        self.doc_emb = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def save(self, path):
        np.save(path, self.doc_emb)

    def load(self, path):
        self.doc_emb = np.load(path)

    def search(self, query, k=10):
        """Top-k (doc_id, cosine score) for one query."""
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.doc_emb @ q               # cosine similarity with every doc
        top = np.argsort(-scores)[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in top]

    def search_many(self, queries, k=10):
        """queries: {query_id: text}. Returns {query_id: [doc_id, ...]}."""
        qids = list(queries)
        Q = self.model.encode([queries[q] for q in qids],
                              batch_size=64, normalize_embeddings=True)
        scores = Q @ self.doc_emb.T             # shape: (num_queries, num_docs)
        results = {}
        for row, qid in enumerate(qids):
            top = np.argsort(-scores[row])[:k]
            results[qid] = [self.doc_ids[i] for i in top]
        return results