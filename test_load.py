from beir.datasets.data_loader import GenericDataLoader

corpus, queries, qrels = GenericDataLoader("data/scifact").load(split="test")
print(len(corpus), len(queries), len(qrels))