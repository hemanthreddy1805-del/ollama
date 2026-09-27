import ollama
import chromadb

# -------------------------
# 1. Create Vector Database
# -------------------------

client = chromadb.Client()

collection = client.create_collection(
    name="rag_demo"
)

# -------------------------
# 2. Our knowledge
# -------------------------

documents = [
    "Python is a programming language used for software development and artificial intelligence.",
    "RAG stands for Retrieval-Augmented Generation. It allows an LLM to use information retrieved from external documents.",
    "Ollama allows developers to run supported language models locally.",
    "ChromaDB is a vector database used to store and search embeddings."
]

# -------------------------
# 3. Store documents
# -------------------------

for i, document in enumerate(documents):

    response = ollama.embed(
        model="nomic-embed-text",
        input=document
    )

    embedding = response["embeddings"][0]

    collection.add(
        ids=[str(i)],
        documents=[document],
        embeddings=[embedding]
    )

# -------------------------
# 4. User question
# -------------------------

question = "What is RAG?"

# -------------------------
# 5. Convert question to vector
# -------------------------

response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

query_embedding = response["embeddings"][0]

# -------------------------
# 6. Search VectorDB
# -------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

context = "\n".join(results["documents"][0])

print("\nRetrieved Information:")
print(context)

# -------------------------
# 7. Give context to LLM
# -------------------------

prompt = f"""
Answer the question using only the information below.

Context:
{context}

Question:
{question}
"""

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nFinal Answer:")
print(response["message"]["content"])