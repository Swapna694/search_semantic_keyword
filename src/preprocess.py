import re
import nltk
from nltk.stem import PorterStemmer

# Stopwords: try NLTK first, fall back to scikit-learn's list if offline
try:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOPWORDS = set(ENGLISH_STOP_WORDS)

STEMMER = PorterStemmer()
TOKEN_RE = re.compile(r"[a-z0-9]+")


def preprocess(text, stem=True, remove_stopwords=True):
    """Convert raw text into a list of cleaned tokens."""
    tokens = TOKEN_RE.findall(text.lower())          # lowercase + tokenize
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [STEMMER.stem(t) for t in tokens]
    return tokens