from fastapi import FastAPI 
from sentence_transformers import SentenceTransformer
from typing import List
from pydantic import BaseModel

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

class TextsRequest(BaseModel):
    texts: List[str]

@app.post("/embed")
async def embed_texts(request: TextsRequest):

    embeddings = model.encode(
        request.texts,
        normalize_embeddings=True
    )

    return {"embeddings": embeddings.tolist()}