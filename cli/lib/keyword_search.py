import os
import string
import pickle
import math
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    BM25_B,
    CACHE_DIR,
    load_movies,
    load_stopwords,
    format_search_result
)


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = dict()
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = dict()
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        
    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self):
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        doc_count = len(self.doc_lengths)
        lengths_sum = sum(self.doc_lengths.values())
        return lengths_sum / doc_count

    def build(self):
        movies = load_movies()
        for movie in movies:
            movie_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], movie_text)
            self.docmap[movie["id"]] = movie
    
    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)
    
    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
    
    def get_documents(self, term):
        return sorted(self.index.get(term, set()))
    
    def get_tf(self, doc_id, term):
        tokenized = tokenize_text(term)
        if len(tokenized) != 1:
            raise Exception("term must be one token")
        return self.term_frequencies[doc_id][tokenized[0]]
    
    def get_idf(self, term):
        tokenized = tokenize_text(term)
        if len(tokenized) != 1:
            raise Exception("term must be one token")
        doc_count = len(self.docmap)
        term_doc_count = len(self.get_documents(tokenized[0]))
        idf = math.log((doc_count + 1) / (term_doc_count + 1))
        return idf
    
    def get_tfidf(self, doc_id, term):
        return self.get_tf(doc_id, term) * self.get_idf(term)
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = (1 - b) + (b * doc_length / avg_doc_length)
        else:
            length_norm = 1
        saturated_tf = (tf * (k1 + 1) / (tf + k1 * length_norm))
        return saturated_tf

    def get_bm25_idf(self, term):
        tokenized = tokenize_text(term)
        if len(tokenized) != 1:
            raise Exception("term must be one token")
        doc_count = len(self.docmap)
        term_doc_count = len(self.get_documents(tokenized[0]))
        bm25_idf = math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
        return bm25_idf
    
    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        query_tokens = tokenize_text(query)
        scores = defaultdict(float)
        for doc_id in self.docmap:
            for qt in query_tokens:
                scores[doc_id] += self.bm25(doc_id, qt)
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"], 
                score=score
            )
            results.append(formatted_result)

        return results

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text):
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = [t for t in tokens if t]
    stopwords = load_stopwords()
    stemmer = PorterStemmer()
    filtered_words = [stemmer.stem(t) for t in valid_tokens if t not in stopwords]
    return filtered_words

def build_command():
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()

def search_command(query, limit=DEFAULT_SEARCH_LIMIT):
    inverted_index = InvertedIndex()
    inverted_index.load()

    query_tokens = tokenize_text(query)
    results = []
    seen = set()
    for qt in query_tokens:
        for doc_id in inverted_index.get_documents(qt):
            if doc_id not in seen:
                results.append(inverted_index.docmap[doc_id])
                seen.add(doc_id)
            if len(results) >= limit:
                return results
    return results

def tf_command(doc_id, term):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_tf(doc_id, term)

def idf_command(term):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_idf(term)

def tfidf_command(doc_id, term):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_tfidf(doc_id, term)

def bm25_idf_command(term):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_idf(term)

def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_tf(doc_id, term, k1, b)

def bm25search_command(query, limit=DEFAULT_SEARCH_LIMIT):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.bm25_search(query, limit)