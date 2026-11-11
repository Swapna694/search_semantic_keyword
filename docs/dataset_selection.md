# Dataset Selection

I looked at three datasets from the BEIR benchmark. I needed one with documents,
queries and relevance labels, so I could try both keyword search and semantic search
and measure the results.

## Comparison

|                         | SciFact             | NFCorpus            | ArguAna        |
|-------------------------|---------------------|---------------------|----------------|
| Domain                  | Scientific papers   | Medical / nutrition | Online debates |
| Documents               | ~5,183              | ~3,600              | ~8,700         |
| Test queries            | 300                 | 323                 | 1,406          |
| Query style             | Full-sentence claim | Short keywords      | Long argument  |
| Relevant docs per query | ~1.1                | ~38                 | 1              |

## Decision: SciFact

I chose SciFact for these reasons:
1. It has documents, queries and relevance labels, so I can measure results.
2. It is small, so TF-IDF, BM25 and embeddings run on my laptop.
3. Abstracts contain exact technical terms (good for keyword search), but claims are
   often worded differently from the abstract (good for semantic search).
4. Queries are full sentences, which work well with embedding models.

## Why not the others
- NFCorpus: queries are very short, so keyword search already works well and the
  comparison is less interesting.
- ArguAna: queries are long passages and the task is finding counter-arguments,
  which is not typical search.

## Data used
BEIR SciFact test split (300 queries, labels from qrels/test.tsv).
My claims_test.csv has the same 300 queries but no labels.