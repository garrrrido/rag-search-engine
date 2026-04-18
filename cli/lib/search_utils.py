import os
import json


DEFAULT_SEARCH_LIMIT = 5
DEFAULT_ALPHA = 0.5
DEFAULT_RRF_K = 60
SEARCH_MULTIPLIER = 5
DOCUMENT_PREVIEW_LENGTH = 100
DEFAULT_CHUNK_SIZE = 200
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 1
SCORE_PRECISION = 3
BM25_K1 = 1.5
BM25_B = 0.75
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")


def load_movies():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
    
def load_golden_dataset() -> dict:
    with open(GOLDEN_DATASET_PATH, "r") as f:
        return json.load(f)

def format_search_result(doc_id, title, document, score, **metadata):
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {}
    }