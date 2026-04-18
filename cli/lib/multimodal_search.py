import os
from PIL import Image
from sentence_transformers import SentenceTransformer
from .search_utils import load_movies
from .semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(self, documents=[], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
    
    def embed_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)
        embedding = self.model.encode([image])[0]
        return embedding
    
    def search_with_image(self, image_path):
        image_emb = self.embed_image(image_path)
        docs_with_cossims = [{
            "id": doc["id"],
            "title": doc["title"],
            "description": doc["description"],
            "similarity": cosine_similarity(text_emb, image_emb)
        } for doc, text_emb in zip(self.documents, self.text_embeddings)]

        docs_with_cossims.sort(key=lambda x: x["similarity"], reverse=True)
        return docs_with_cossims[:5]
        

def verify_image_embedding(image_path):
    multimodal_instance = MultimodalSearch()
    embedding = multimodal_instance.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):
    movies = load_movies()
    multimodal_instance = MultimodalSearch(movies)
    results = multimodal_instance.search_with_image(image_path)
    return results