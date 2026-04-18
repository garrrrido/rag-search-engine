import os
import time
import json
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-4-31b-it"
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

def rerank(query, documents, method="batch", limit=5):
    match method:
        case "individual":
            return llm_rerank_individual(query, documents, limit)
        case "batch":
            return llm_rerank_batch(query, documents, limit)
        case "cross_encoder":
            return cross_encoder_rerank(query, documents, limit)
        case _:
            return documents[:limit]

def llm_rerank_individual(query, documents, limit):
    scored_docs = []
    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score: 
"""
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        score_text = (response.text or "").strip()
        scored_docs.append({
            **doc,
            "individual_score": int(score_text)
        })
        time.sleep(5)
    
    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]

def llm_rerank_batch(query, documents, limit):
    if not documents:
        return []

    doc_str_list = [f"{doc['id']}: {doc['title']} - {doc['document'][:200]}..." for doc in documents]
    doc_str = "\n".join(doc_str_list)
    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_str}

Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 34, 2, 1]

Ranking:
"""
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    ranked_ids = json.loads(response.text.strip())
    reranked = [{
        **doc,
        "batch_rank": ranked_ids.index(doc["id"]) + 1
    } for doc in documents if doc["id"] in ranked_ids]
    reranked.sort(key=lambda x: x["batch_rank"])

    return reranked[:limit]

def cross_encoder_rerank(query, documents, limit):
    pairs = [[query, f"{doc.get('title', '')} - {doc.get('document', '')}"] for doc in documents]
    scores = cross_encoder.predict(pairs)
    scored_docs = [{
        **doc,
        "crossencoder_score": score
    } for doc, score in zip(documents, scores)]
    scored_docs.sort(key=lambda x: x["crossencoder_score"], reverse=True)
    return scored_docs[:limit]