import argparse
from lib.evaluation import llm_judge_results
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_ALPHA,
    DEFAULT_RRF_K
)
from lib.hybrid_search import (
    normalize_scores,
    weighted_search_command,
    rrf_search_command
)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="List of scores to normalize")

    weighted_parser = subparsers.add_parser("weighted-search", help="Perform hybrid search")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25)")
    weighted_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to return")

    rrf_parser = subparsers.add_parser("rrf-search", help="Perform rrf search")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=DEFAULT_RRF_K, help="K parameter for rrf search")
    rrf_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to return")
    rrf_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="LLM query enhancement")
    rrf_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Method for reranking")
    rrf_parser.add_argument("--evaluate", action="store_true", help="Use LLM to evaluate result relevance")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_scores(args.scores)
            for ns in normalized_scores:
                print(f"* {ns:.4f}")
        case "weighted-search":
            result = weighted_search_command(args.query, args.alpha, args.limit)
            print(f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):")
            print(f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic")
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}")
                print(f"   {res['document'][:100]}...\n")
        case "rrf-search":
            result = rrf_search_command(args.query, args.k, args.enhance, args.rerank_method, args.limit)
            if result["enhanced_query"]:
                print(f"Enhanced query ({result['enhance_method']}): '{result['original_query']}' -> '{result['enhanced_query']}'\n")
            if result["reranked"]:
                print(f"Re-ranking top {len(result['results'])} results using {result['rerank_method']} method...\n")

            print(f"Reciprocal Rank Fusion Results for '{result['query']}' (k={result['k']}):")
            for i, res in enumerate(result["results"]):
                print(f"{i+1}. {res['title']}")
                if "individual_score" in res:
                    print(f"   Re-rank Score: {res.get('individual_score', 0):.3f}/10")
                if "batch_rank" in res:
                    print(f"   Re-rank Rank: {res.get('batch_rank', 0)}")
                if "crossencoder_score" in res:
                    print(f"   Cross Encoder Score: {res.get('crossencoder_score', 0):.3f}")
                    
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                ranks = []
                if metadata.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
                if metadata.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   {res['document'][:100]}...\n")
            if args.evaluate:
                print("LLM Evaluation (0-3 relevance scale):")
                llm_scores = llm_judge_results(args.query, result["results"])
                for i, (res, score) in enumerate(zip(result["results"], llm_scores)):
                    print(f"{i+1}. {res['title']}: {score}/3")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()