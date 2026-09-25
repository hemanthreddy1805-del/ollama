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
-experience
- technology
- skill
-goal
"""
        },
        {
            "role": "user",
            "content": "My name is Hemanth. i have 3 years of experience in software field.i know python and sql .and i want to became an gen ai engineer"
        }
    ]
)

answer = response["message"]["content"]

print(answer)