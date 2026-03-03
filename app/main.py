from fastapi import FastAPI
from app.schemas import SearchRequest
from app.models import load_csv_data, search_documents
from app.embedding import generate_embedding

app = FastAPI()

@app.on_event("startup")
def startup_event():
    load_csv_data("url_documents.csv")
    print("CSV vectors loaded")

@app.post("/vector-search")
async def vector_search(request: SearchRequest):
    query_vector = generate_embedding(request.text)
    results = search_documents(query_vector)
    return {"results": results}