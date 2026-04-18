import argparse
from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.augmented_generation import (
    rag_command,
    summarize_command,
    citations_command,
    question_command
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Perform Hybrid search and summarize the results")
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to summarize")

    citations_parser = subparsers.add_parser("citations", help="Perform Hybrid search while listing the sources used")
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum documents to use")

    question_parser = subparsers.add_parser("question", help="Answer a question")
    question_parser.add_argument("question", type=str, help="Question to ask")
    question_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum documents to use")

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print("\nRAG Response:")
            print(result["answer"])
        case "summarize":
            result = summarize_command(args.query, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print("\nLLM Summary:")
            print(result["summary"])
        case "citations":
            result = citations_command(args.query, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print("\nLLM Answer:")
            print(result["answer"])
        case "question":
            result = question_command(args.question, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print("\nAnswer:")
            print(result["answer"])
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()