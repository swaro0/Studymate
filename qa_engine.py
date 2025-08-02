from backend.embeddings import model
import faiss
import numpy as np
import requests

WATSONX_API_KEY = "your_watsonx_key"
FAISS_INDEX_PATH = "models/faiss_index/index.faiss"

def load_index():
    return faiss.read_index(FAISS_INDEX_PATH)

def search_chunks(query, chunks, top_k=3):
    query_embedding = model.encode([query])
    index = load_index()
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [chunks[i] for i in indices[0]]

def generate_answer(query, context):
    prompt = f"""
    You are StudyMate AI. Answer the question using the context below:
    Context: {context}
    Question: {query}
    Answer:
    """
    # Placeholder for Watsonx call
    return "Answer generation placeholder. Integrate Watsonx API here."
