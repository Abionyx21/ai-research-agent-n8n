import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

st.title("🧠 Local RAG Chatbot (TinyLlama)")

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load knowledge
with open("data/knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = [text[i:i+500] for i in range(0, len(text), 500)]
embeddings = embed_model.encode(chunks)

# Vector DB
client = chromadb.Client()
collection = client.get_or_create_collection("rag")

for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        embeddings=[embeddings[i].tolist()],
        ids=[str(i)]
    )

query = st.text_input("Ask a question")

if query:
    query_embedding = embed_model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = " ".join(results["documents"][0])

    response = ollama.generate(
        model="tinyllama",
        prompt=f"Answer using this context:\n{context}\n\nQuestion: {query}"
    )

    st.write("### Answer")
    st.write(response["response"])