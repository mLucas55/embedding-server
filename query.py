print("Initializing client...")
import requests
import chromadb

# Use PersistentClient for persistence
client = chromadb.PersistentClient(path="./chroma_db")

# Try to get the collection, create if it doesn't exist
try:
    collection = client.get_collection("my_collection")
except chromadb.errors.NotFoundError:
    pass

url = "http://127.0.0.1:8000/embed"


data = ["fruit"]

response = requests.post(url, json={'texts': data})

embedding = response.json()["embeddings"]


results = collection.query(
    query_embeddings=embedding,
    n_results=5
)

ids = results["ids"][0]
documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

for i, (id_, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances), start=1):
    print(f"Result {i}")
    print(f"  ID       : {id_}")
    print(f"  Document : {doc}")
    print(f"  Metadata : {meta}")
    print(f"  Distance : {dist:.4f}")
    print()