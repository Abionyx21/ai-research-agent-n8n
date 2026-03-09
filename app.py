import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Read knowledge file
with open("data/knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split into chunks
chunks = text.split("\n")

# Create embeddings
embeddings = embed_model.encode(chunks)

# Create ChromaDB
client = chromadb.Client()
collection = client.create_collection(name="rag")

# Store documents
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        embeddings=[embeddings[i].tolist()],
        ids=[str(i)]
    )

# Ask question
query = input("Ask a question: ")

# Embed query
query_embedding = embed_model.encode([query])[0].tolist()

# Retrieve context
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

context = " ".join(results["documents"][0])

# Send to TinyLlama
response = ollama.generate(
    model="tinyllama",
    prompt=f"Answer using this context:\n{context}\n\nQuestion: {query}"
)

print("\nAnswer:\n")
print(response["response"])