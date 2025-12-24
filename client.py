print("Initializing client...")
import requests
import chromadb
from chromadb.config import Settings

# Use PersistentClient for persistence
client = chromadb.PersistentClient(path="./chroma_db")

# Try to get the collection, create if it doesn't exist
try:
    collection = client.get_collection("my_collection")
except chromadb.errors.NotFoundError:
    collection = client.create_collection("my_collection")

url = "http://127.0.0.1:8000/embed"

data = ["apple", "gun", "bomb", "strawberry", "China"]

print("Posting to server")
response = requests.post(url, json={'texts': data})

embeddings = response.json()["embeddings"]

print("Adding to collection")
collection.add(
    ids=["id1", "id2", "id3", "id4", "id5"],
    documents=data,
    embeddings=embeddings
)
print("Added!")