import ollama
import json

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "system",
            "content": """
You are an AI assistant.

Return the answer ONLY as valid JSON.

The JSON must contain:
- name
- role
- skill
"""
        },
        {
            "role": "user",
            "content": "My name is Hemanth. I am learning Generative AI and Python."
        }
    ]
)

answer = response["message"]["content"]

print(answer)