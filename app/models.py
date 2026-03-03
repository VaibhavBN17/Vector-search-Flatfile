import csv
import json
import numpy as np

documents = []
embeddings_matrix = None

def load_csv_data(filepath: str):
    global documents, embeddings_matrix

    docs = []
    vectors = []

    with open(filepath, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append({
                "id": row["id"],
                "url": row["url"],
                "content": row["content"]
            })

            embedding = np.array(json.loads(row["embedding"]), dtype=np.float32)
            vectors.append(embedding)

    documents = docs
    embeddings_matrix = np.vstack(vectors)

    # Normalize once for cosine similarity
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    embeddings_matrix /= norms

def search_documents(query_embedding: np.ndarray, top_k: int = 10):
    global embeddings_matrix

    # Normalize query
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Cosine similarity (VERY FAST)
    scores = np.dot(embeddings_matrix, query_embedding)

    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "id": documents[idx]["id"],
            "url": documents[idx]["url"],
            "content": documents[idx]["content"]
        })

    return results