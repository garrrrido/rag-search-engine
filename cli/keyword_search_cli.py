import argparse
from lib.search_utils import BM25_K1, BM25_B
from lib.keyword_search import (
    search_command, 
    build_command, 
    tf_command, 
    idf_command, 
    tfidf_command,
    bm25_idf_command,
    bm25_tf_command,
    bm25search_command
)


def main():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the index and docmap")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a given document ID and term")
    tf_parser.add_argument("doc_id", type=int, help="Doc ID to find frequency in")
    tf_parser.add_argument("term", type=str, help="Term to find frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a given term")
    idf_parser.add_argument("term", type=str, help="Term to find idf for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get tf-idf for a given document ID and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Doc ID to find tf-idf in")
    tfidf_parser.add_argument("term", type=str, help="Term to find tf-idf for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get bm25 idf for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to find bm25 idf for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get bm25 tf score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Doc ID to find bm25 tf in")
    bm25_tf_parser.add_argument("term", type=str, help="Term to find bm25 tf for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="K1 parameter for bm25")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="B parameter for bm25")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "build":
            print("Building the inverted index...")
            build_command()
            print("Inverted index built successfully")
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document {args.doc_id}: {tf}")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document {args.doc_id}: {tfidf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document {args.doc_id}: {bm25tf:.2f}")
        case "bm25search":
            print("Searching for:", args.query)
            results = bm25search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']} - Score: {res['score']:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
