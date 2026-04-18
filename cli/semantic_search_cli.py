import argparse
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP
)
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    semantic_search,
    chunk_text,
    semantic_chunk_text,
    embed_chunks_command,
    search_chunked_command
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify and print the model")

    single_embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a single text")
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify the embeddings of the movie dataset")

    embed_query_parser = subparsers.add_parser("embedquery", help="Generate an embedding for a query")
    embed_query_parser.add_argument("query", type=str, help="Text to embed")

    search_parser = subparsers.add_parser("search", help="Semantic search for movies")
    search_parser.add_argument("query", type=str, help="Search Query")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Max number of results")

    search_parser = subparsers.add_parser("chunk", help="Split text into fixed-size chunks")
    search_parser.add_argument("text", type=str, help="Text to chunk")
    search_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Max number of words per chunk")
    search_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Number of words to overlap between chunks")

    search_parser = subparsers.add_parser("semantic_chunk", help="Split text into semantic chunks")
    search_parser.add_argument("text", type=str, help="Text to chunk")
    search_parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_SEMANTIC_CHUNK_SIZE, help="Max number of each chunk in sentences")
    search_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Number of sentences to overlap between chunks")

    subparsers.add_parser("embed_chunks", help="Generate embeddings for chunked documents")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search using chunked embeddings")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embeddings = embed_chunks_command()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            result = search_chunked_command(args.query, args.limit)
            print(f"Query: {result['query']}")
            print("Results:")
            for i, res in enumerate(result["results"], 1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()