import ollama

response = ollama.embed(
    model="nomic-embed-text",
    input="Python is a programming language"
)

embedding = response["embeddings"][0]

print("Number of values:", len(embedding))
print("First 10 values:", embedding[:10])