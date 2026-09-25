import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "system",
            "content": "expain me what is azure data factor and azure data bricks and sql"
        },
        {
            "role": "user",
            "content": "What is a Python function?"
        }
    ]
)

print(response["message"]["content"])