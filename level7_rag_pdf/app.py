import os
import ast
import operator
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

    try:

        client.delete_collection(
            "company_documents"
        )

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
# 9. SAFE CALCULATOR
# ==================================================

def calculate(expression):
    """
    Calculator tool.

    Supports:
    +  -  *  /  %  **
    parentheses
    """

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos
    }

    def evaluate(node):

        # Number
        if isinstance(node, ast.Constant):

            if isinstance(node.value, (int, float)):

                return node.value

            raise ValueError(
                "Only numbers are allowed."
            )

        # Binary operation
        elif isinstance(node, ast.BinOp):

            left = evaluate(node.left)
            right = evaluate(node.right)

            operation = allowed_operators.get(
                type(node.op)
            )

            if operation is None:

                raise ValueError(
                    "Operator not allowed."
                )

            return operation(left, right)

        # Positive / negative number
        elif isinstance(node, ast.UnaryOp):

            value = evaluate(node.operand)

            operation = allowed_operators.get(
                type(node.op)
            )

            if operation is None:

                raise ValueError(
                    "Operator not allowed."
                )

            return operation(value)

        else:

            raise ValueError(
                "Invalid mathematical expression."
            )

    try:

        tree = ast.parse(
            expression,
            mode="eval"
        )

        result = evaluate(tree.body)

        return str(result)

    except Exception as e:

        return f"Calculation error: {str(e)}"


# ==================================================
# 10. CALCULATOR TOOL DEFINITION
# ==================================================

calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Use this tool when a mathematical "
            "calculation is required. "
            "Examples: 25% of 80000, "
            "100 + 200, 500 * 20, "
            "(1000 + 500) / 2."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Mathematical expression "
                        "to calculate."
                    )
                }
            },
            "required": ["expression"]
        }
    }
}


# ==================================================
# 11. RAG SEARCH
# ==================================================

def search_documents(collection, question):

    question_embedding = create_embedding(
        question
    )

    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=5
    )

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

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

    return context


# ==================================================
# 12. ASK QUESTION WITH TOOL CALLING
# ==================================================

def ask_question(collection, question):

    # ------------------------------------------------
    # First perform RAG search
    # ------------------------------------------------

    context = search_documents(
        collection,
        question
    )

    # ------------------------------------------------
    # System instruction
    # ------------------------------------------------

    system_prompt = """
You are a company knowledge assistant.

You have access to a calculator tool.

Rules:

1. Use the provided document context to answer
   questions about company documents.

2. If the user asks for a mathematical calculation,
   use the calculator tool.

3. Never calculate complicated mathematical
   expressions yourself when the calculator tool
   can be used.

4. Do not invent information.

5. If the information cannot be found in the
   documents, clearly say so.

6. Always mention the source file when possible.

7. Give the final answer in a simple and clear way.
"""

    # ------------------------------------------------
    # Initial messages
    # ------------------------------------------------

    messages = [

        {
            "role": "system",
            "content": system_prompt
        },

        {
            "role": "user",
            "content": f"""
Context from company documents:

---------------- CONTEXT ----------------

{context}

---------------- QUESTION ----------------

{question}
"""
        }
    ]

    # ------------------------------------------------
    # First LLM call
    # ------------------------------------------------

    response = ollama.chat(

        model=LLM_MODEL,

        messages=messages,

        tools=[
            calculator_tool
        ]
    )

    # ------------------------------------------------
    # Check whether LLM requested a tool
    # ------------------------------------------------

    if response.message.tool_calls:

        # Add LLM response to conversation
        messages.append(response.message)

        # Process every tool call
        for tool_call in response.message.tool_calls:

            if tool_call.function.name == "calculate":

                expression = (
                    tool_call.function.arguments[
                        "expression"
                    ]
                )

                print(
                    f"\n[Calculator Tool Called]"
                )

                print(
                    f"Expression: {expression}"
                )

                result = calculate(
                    expression
                )

                print(
                    f"Result: {result}"
                )

                # Send calculator result
                # back to the LLM

                messages.append({

                    "role": "tool",

                    "content": (
                        f"Calculator result: "
                        f"{result}"
                    )
                })

        # ------------------------------------------------
        # Ask LLM for final answer
        # ------------------------------------------------

        final_response = ollama.chat(

            model=LLM_MODEL,

            messages=messages,

            tools=[
                calculator_tool
            ]
        )

        return final_response.message.content

    # ------------------------------------------------
    # No tool required
    # ------------------------------------------------

    return response.message.content


# ==================================================
# MAIN PROGRAM
# ==================================================

print("\n====================================")
print("RAG + CALCULATOR TOOL")
print("====================================")


# --------------------------------------------------
# Load files
# --------------------------------------------------

documents = load_all_files()

print(
    f"\nTotal files loaded: {len(documents)}"
)


# --------------------------------------------------
# Create chunks
# --------------------------------------------------

chunks = create_chunks(documents)

print(
    f"Total chunks created: {len(chunks)}"
)


# --------------------------------------------------
# Create vector database
# --------------------------------------------------

collection = create_vector_database(
    chunks
)


print("\n====================================")
print("RAG SYSTEM READY")
print("CALCULATOR TOOL READY")
print("====================================")


# ==================================================
# CHAT LOOP
# ==================================================

while True:

    question = input(
        "\nAsk your question "
        "(type 'exit' to stop): "
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