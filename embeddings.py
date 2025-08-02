from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_faiss_index(chunks, index_path="models/faiss_index/index.faiss"):
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    
    if not os.path.exists("models/faiss_index"):
        os.makedirs("models/faiss_index")
    faiss.write_index(index, index_path)
    
    return index, embeddings
