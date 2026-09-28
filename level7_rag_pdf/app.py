import os
import pandas as pd
from pypdf import PdfReader
from docx import Document
import ollama
import chromadb


# ==================================================
# SETTINGS
# ==================================================

DATA_FOLDER = "data"

LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"


# ==================================================
# 1. PDF LOADER
# ==================================================

def load_pdf(path):

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==================================================
# 2. DOCX LOADER
# ==================================================

def load_docx(path):

    document = Document(path)

    text = ""

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            text += paragraph.text + "\n"

    return text


# ==================================================
# 3. EXCEL LOADER
# ==================================================

def load_excel(path):

    df = pd.read_excel(path)

    return df.to_string(index=False)


# ==================================================
# 4. CSV LOADER
# ==================================================

def load_csv(path):

    df = pd.read_csv(path)

    return df.to_string(index=False)


# ==================================================
# 5. LOAD ALL FILES
# ==================================================

def load_all_files():

    documents = []

    for filename in os.listdir(DATA_FOLDER):

        path = os.path.join(DATA_FOLDER, filename)

        if filename.lower().endswith(".pdf"):

            text = load_pdf(path)

        elif filename.lower().endswith(".docx"):

            text = load_docx(path)

        elif filename.lower().endswith(".xlsx"):

            text = load_excel(path)

        elif filename.lower().endswith(".csv"):

            text = load_csv(path)

        else:

            continue

        documents.append({
            "source": filename,
            "text": text
        })

        print(f"Loaded: {filename}")

    return documents


# ==================================================
# 6. CHUNKING
# ==================================================

def create_chunks(documents, chunk_size=1000, overlap=200):

    chunks = []

    for document in documents:

        text = document["text"]
        source = document["source"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end]

            if chunk.strip():

                chunks.append({
                    "text": chunk,
                    "source": source
                })

            start += chunk_size - overlap

    return chunks


# ==================================================
# 7. CREATE EMBEDDING
# ==================================================

def create_embedding(text):

    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )

    return response["embedding"]


# ==================================================
# 8. CREATE VECTOR DATABASE
# ==================================================

def create_vector_database(chunks):

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    # Delete old collection if it exists
    try:

        client.delete_collection("company_documents")

    except Exception:

        pass

    collection = client.create_collection(
        name="company_documents"
    )

    print("\nCreating embeddings...\n")

    for i, chunk in enumerate(chunks):

        embedding = create_embedding(
            chunk["text"]
        )

        collection.add(

            ids=[str(i)],

            embeddings=[embedding],

            documents=[chunk["text"]],

            metadatas=[
                {
                    "source": chunk["source"]
                }
            ]
        )

        print(
            f"Embedded chunk {i + 1}/{len(chunks)}"
        )

    return collection


# ==================================================
# 9. ASK QUESTION
# ==================================================

def ask_question(collection, question):

    # Create embedding for user question

    question_embedding = create_embedding(
        question
    )

    # Search vector database

    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=5
    )

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]


    # Combine retrieved information

    context_parts = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        source = metadata["source"]

        context_parts.append(
            f"Source: {source}\n"
            f"{document}"
        )

    context = "\n\n----------------\n\n".join(
        context_parts
    )


    # ==================================================
    # PROMPT FOR LLM
    # ==================================================

    prompt = f"""
You are a company knowledge assistant.

Answer the user's question using ONLY the
information provided in the context.

The context can come from PDF, DOCX, Excel,
or CSV files.

If the answer cannot be found in the context,
say:

"I could not find this information in the
provided documents."

Do not invent information.

Always mention the source file when possible.

---------------- CONTEXT ----------------

{context}

---------------- QUESTION ----------------

{question}

---------------- ANSWER ----------------
"""


    # ==================================================
    # CALL OLLAMA
    # ==================================================

    response = ollama.chat(

        model=LLM_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ==================================================
# MAIN PROGRAM
# ==================================================

print("\n====================================")
print("MULTI-FORMAT RAG SYSTEM")
print("====================================")


# Load files

documents = load_all_files()

print(
    f"\nTotal files loaded: {len(documents)}"
)


# Create chunks

chunks = create_chunks(documents)

print(
    f"Total chunks created: {len(chunks)}"
)


# Create vector database

collection = create_vector_database(
    chunks
)


print("\n====================================")
print("RAG SYSTEM READY")
print("====================================")


# ==================================================
# CHAT LOOP
# ==================================================

while True:

    question = input(
        "\nAsk your question (type 'exit' to stop): "
    )

    if question.lower() == "exit":

        print("Goodbye!")

        break


    answer = ask_question(
        collection,
        question
    )


    print("\n====================================")
    print("ANSWER")
    print("====================================")

    print(answer)