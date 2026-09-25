import ollama

response = ollama.embed(
    model="nomic-embed-text",
    input="my name is hemanth reddy am learning the gen ai right now"
)

embedding = response["embeddings"][0]

print("Number of values:", len(embedding))
print("First 10 values:", embedding[:10])