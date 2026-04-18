import os
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DOCUMENT_PREVIEW_LENGTH,
    load_movies,
    format_search_result
)


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = dict()
        self.movie_embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
    
    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        movie_list = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_list.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_list, show_progress_bar=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.movie_embeddings_path, self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.movie_embeddings_path):
            self.embeddings = np.load(self.movie_embeddings_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")
        
        query_emb = self.generate_embedding(query)
        cs_scores = [(cosine_similarity(query_emb, emb), self.documents[idx]) for idx, emb in enumerate(self.embeddings)]
        cs_scores.sort(key=lambda x: x[0], reverse=True)
        return [{
            "score": score,
            "title": doc["title"],
            "description": doc["description"]
        } for score, doc in cs_scores[:limit]]
    

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")
    
    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        all_chunks, all_metadata = [], []
        for idx, doc in enumerate(documents):
            if not doc.get("description", "").strip():
                continue
            chunks = semantic_chunk(doc["description"], DEFAULT_SEMANTIC_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
            all_chunks.extend(chunks)
            all_metadata.extend([{
                "movie_idx": idx,
                "chunk_idx": cidx,
                "total_chunks": len(chunks)
            } for cidx, c in enumerate(chunks)])

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = all_metadata

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": all_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings
            
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query, limit=10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("No chunk embeddings loaded. Call load_or_create_chunk_embeddings first.")
        
        query_emb = self.generate_embedding(query)
        chunk_scores = [{
            "chunk_idx": i,
            "movie_idx": self.chunk_metadata[i]["movie_idx"],
            "score": cosine_similarity(query_emb, ch_emb)
        } for i, ch_emb in enumerate(self.chunk_embeddings)]

        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (movie_idx not in movie_scores or chunk_score["score"] > movie_scores[movie_idx]):
                movie_scores[movie_idx] = chunk_score["score"]
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for movie_idx, score in sorted_movies[:limit]:
            if movie_idx is None:
                continue
            doc = self.documents[movie_idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score=score,
                )
            )
        return results
            

def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")

def embed_text(text):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search(query, limit):
    documents = load_movies()
    search_instance = SemanticSearch()
    search_instance.load_or_create_embeddings(documents)
    results = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:\n")
    for i, res in enumerate(results):
        print(f"{i+1}. {res['title']} (score: {res['score']:.4f})")
        if len(res['description']) >= 100:
            print(f"   {res['description'][:100]}...\n")
        else:
            print(f"   {res['description']}\n")

def fixed_size_chunking(text, chunk_size, overlap):
    words = text.split()

    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks

def chunk_text(text, chunk_size, overlap):
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")

def semantic_chunk(text, max_chunk_size, overlap):
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        return [text]

    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i:i+max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        
        stripped_sentences = [s.strip() for s in chunk_sentences if s.strip()]
        if stripped_sentences:
            chunks.append(" ".join(stripped_sentences))
        i += max_chunk_size - overlap

    return chunks
    
def semantic_chunk_text(text, max_chunk_size, overlap):
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def embed_chunks_command():
    movies = load_movies()
    search_instance = ChunkedSemanticSearch()
    return search_instance.load_or_create_chunk_embeddings(movies)

def search_chunked_command(query, limit):
    documents = load_movies()
    search_instance = ChunkedSemanticSearch()
    search_instance.load_or_create_chunk_embeddings(documents)
    results = search_instance.search_chunks(query, limit)
    return {
        "query": query,
        "results": results
    }