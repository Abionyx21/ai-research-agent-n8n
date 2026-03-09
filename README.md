# Local RAG Chatbot using TinyLlama

A Retrieval-Augmented Generation (RAG) chatbot built using TinyLlama, ChromaDB, and Streamlit that retrieves relevant information from documents and generates contextual answers locally.

## Tech Stack
- TinyLlama (Ollama)
- ChromaDB
- Sentence Transformers
- Streamlit
- Python

## Project Architecture

User Query
↓
Embedding Model
↓
Vector Search (ChromaDB)
↓
Relevant Context Retrieved
↓
TinyLlama Generates Answer
↓
Response to User

## Project Structure

rag-ollama-tinyllama/
│
├── app.py
├── ui.py
├── requirements.txt
├── data/
└── vectordb/

## Installation

Clone the repository

git clone https://github.com/Abionyx21/ai-research-agent-n8n.git

Install dependencies

pip install -r requirements.txt

Install Ollama and run TinyLlama

ollama run tinyllama

Run the chatbot

streamlit run ui.py

## Example

User Question:
What is AI used for?

Answer:
AI is used in healthcare, finance, and education for tasks like medical diagnosis, fraud detection, and personalized learning.
