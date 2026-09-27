import chromadb
import ollama

client = chromadb.Client()

collection = client.create_collection(
    name="genai_learning"
)

documents = [
    "Python is a programming language.",
    "Python is widely used in artificial intelligence and data science.",
    "Java is an object-oriented programming language.",
    "Bangalore is a major technology city in India."
]

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

print("Documents stored successfully!")

query = "Which language is used for AI?"

response = ollama.embed(
    model="nomic-embed-text",
    input=query
)

query_embedding = response["embeddings"][0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\nSearch Results:")

for document in results["documents"][0]:
    print(document)