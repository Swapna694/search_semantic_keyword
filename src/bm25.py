import math
from collections import Counter


class BM25:
    def __init__(self, tokenized_docs, doc_ids, k1=1.5, b=0.75):
        self.doc_ids = doc_ids
        self.k1 = k1
        self.b = b
        self.N = len(tokenized_docs)

        # term counts per document, and each document's length
        self.tf = [Counter(doc) for doc in tokenized_docs]
        self.doc_len = [len(doc) for doc in tokenized_docs]
        self.avg_len = sum(self.doc_len) / self.N

        # document frequency: in how many documents does each word appear?
        df = Counter()
        for doc in tokenized_docs:
            df.update(set(doc))

        # IDF: rare words get a higher value
        self.idf = {
            word: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for word, n in df.items()
        }

    def score(self, query_tokens, i):
        """BM25 score of document number i for the query."""
        total = 0.0
        for word in query_tokens:
            if word not in self.idf:
                continue                       # word never appears in the corpus
            tf = self.tf[i].get(word, 0)
            if tf == 0:
                continue                       # word not in this document
            length_part = 1 - self.b + self.b * self.doc_len[i] / self.avg_len
            total += self.idf[word] * tf * (self.k1 + 1) / (tf + self.k1 * length_part)
        return total

    def search(self, query_tokens, k=10):
        scores = [(self.doc_ids[i], self.score(query_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]