try:
    import docx
except ImportError as exc:
    raise ImportError(
        "The 'python-docx' package is required. Install it with: pip install python-docx"
    ) from exc
import chromadb
import ollama


# -------------------------
# 1. Read DOCX
# -------------------------

doc = docx.Document("document.docx")

text = ""
for para in doc.paragraphs:
    if para.text.strip():
        text += para.text.strip() + "\n"


# -------------------------
# 2. Split into chunks
# -------------------------

chunk_size = 1000

chunks = []

for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

print("Number of chunks:", len(chunks))


# -------------------------
# 3. Create ChromaDB
# -------------------------

client = chromadb.Client()

collection = client.create_collection(
    name="pdf_knowledge"
)


# -------------------------
# 4. Create embeddings
# -------------------------

for i, chunk in enumerate(chunks):

    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )

    embedding = response["embeddings"][0]

    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embedding]
    )


print("Document stored in vector database.")


# -------------------------
# 5. Ask question
# -------------------------

question = input("\nAsk a question: ")


# -------------------------
# 6. Embed question
# -------------------------

response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

query_embedding = response["embeddings"][0]


# -------------------------
# 7. Search ChromaDB
# -------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)


# -------------------------
# 8. Get relevant chunks
# -------------------------

context = "\n\n".join(
    results["documents"][0]
)


# -------------------------
# 9. Create prompt
# -------------------------

prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say "I could not find the answer in the document."
"""


# -------------------------
# 10. Ask LLM
# -------------------------

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# -------------------------
# 11. Print answer
# -------------------------

answer = response["message"]["content"]

print("\nAI:", answer)