from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load document
loader = TextLoader("data/knowledge.txt")
documents = loader.load()

# Split document
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OllamaEmbeddings(model="tinyllama")

# Vector store
vectorstore = Chroma.from_documents(docs, embeddings)

# Load LLM
llm = Ollama(model="tinyllama")

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Ask question
query = input("Ask a question: ")
result = qa_chain.run(query)

print("\nAnswer:\n")
print(result)