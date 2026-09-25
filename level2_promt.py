import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "system",
            "content": "You are a Python teacher. Explain concepts in very simple language and always give a small example."
        },
        {
            "role": "user",
            "content": "What is a Python function?"
        }
    ]
)

print(response["message"]["content"])