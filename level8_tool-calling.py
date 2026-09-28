import ollama


def add_numbers(a, b):
    return a + b


def multiply_numbers(a, b):
    return a * b


tools = [add_numbers, multiply_numbers]

messages = [{"role": "user", "content": "What is 25 + 35?"}]

response = ollama.chat(
    model="llama3.2",
    messages=messages,
    tools=tools
)

messages.append(response.message)

available = {
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers,
}

if response.message.tool_calls:
    for call in response.message.tool_calls:
        func = available[call.function.name]
        result = func(**call.function.arguments)
        print(f"Tool called: {call.function.name}({call.function.arguments}) -> {result}")

        messages.append({
            "role": "tool",
            "tool_name": call.function.name,
            "content": str(result),
        })

    final = ollama.chat(model="llama3.2", messages=messages)
    print("\nAI:", final.message.content)
else:
    print("\nAI:", response.message.content)