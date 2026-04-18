import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .query_enhancement import enhance_query
from .reranking import rerank
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    SEARCH_MULTIPLIER,
    format_search_result,
    load_movies
)

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents

        # semantic search
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        # keyword search
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)
        combined = combine_search_results(bm25_results, semantic_results, alpha)
        return combined[:limit]

    def rrf_search(self, query, k, limit):
        bm25_results = self._bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)
        combined = reciprocal_rank_fusion(bm25_results, semantic_results, k)
        return combined[:limit]
    
def normalize_scores(scores):
    if not scores:
        return []
    
    min_s = min(scores)
    max_s = max(scores)
    if min_s == max_s:
        return [1.0 for i in range(len(scores))]
    
    return [(s - min_s) / (max_s - min_s) for s in scores]

def normalize_search_results(results):
    scores = [result["score"] for result in results]
    normalized = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]
    return results

def hybrid_score(bm25_score, semantic_score, alpha):
    return alpha*bm25_score + (1-alpha)*semantic_score

def rrf_score(rank, k):
    return 1 / (k + rank)

def combine_search_results(bm25_results, semantic_results, alpha):
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}
    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]
    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = [format_search_result(
        doc_id=doc_id,
        title=data["title"],
        document=data["document"],
        score=hybrid_score(data["bm25_score"], data["semantic_score"], alpha),
        bm25_score=data["bm25_score"],
        semantic_score=data["semantic_score"]
    ) for doc_id, data in combined_scores.items()]

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)

def reciprocal_rank_fusion(bm25_results, semantic_results, k):
    rrf_scores = {}
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None
            }
        if rrf_scores[doc_id]["bm25_rank"] is None:
            rrf_scores[doc_id]["bm25_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None
            }
        if rrf_scores[doc_id]["semantic_rank"] is None:
            rrf_scores[doc_id]["semantic_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)

    rrf_results = [format_search_result(
        doc_id=doc_id,
        title=data["title"],
        document=data["document"],
        score=data["rrf_score"],
        rrf_score=data["rrf_score"],
        bm25_rank=data["bm25_rank"],
        semantic_rank=data["semantic_rank"]
    ) for doc_id, data in sorted_items]

    return rrf_results

def weighted_search_command(query, alpha, limit):
    movies = load_movies()
    searcher = HybridSearch(movies)
    return {
        "original_query": query,
        "query": query,
        "alpha": alpha,
        "results": searcher.weighted_search(query, alpha, limit)
    }

def rrf_search_command(query, k, enhance=None, rerank_method=None, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query
    
    search_limit = limit * SEARCH_MULTIPLIER if rerank_method else limit
    results = searcher.rrf_search(query, k, search_limit)

    reranked = False
    if rerank_method:
        results = rerank(query, results, method=rerank_method, limit=limit)
        reranked = True

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "query": query,
        "k": k,
        "rerank_method": rerank_method,
        "reranked": reranked,
        "results": results
    }