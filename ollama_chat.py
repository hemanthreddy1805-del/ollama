import ollama

print("===== MY LOCAL AI =====")

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("AI: Goodbye!")
        break

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    print("\nAI:", response["message"]["content"])