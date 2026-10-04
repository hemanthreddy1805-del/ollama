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
# DATA STORAGE
# ==================================================

# We keep Excel/CSV files as DataFrames here.
#
# This is important because RAG needs text,
# but calculations need structured data.

dataframes = {}


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

    filename = os.path.basename(path)

    # Keep DataFrame for calculations
    dataframes[filename] = df

    # Convert to text for RAG
    text = df.to_string(index=False)

    return text


# ==================================================
# 4. CSV LOADER
# ==================================================

def load_csv(path):

    df = pd.read_csv(path)

    filename = os.path.basename(path)

    # Keep DataFrame for calculations
    dataframes[filename] = df

    # Convert to text for RAG
    text = df.to_string(index=False)

    return text


# ==================================================
# 5. LOAD ALL FILES
# ==================================================

def load_all_files():

    documents = []

    for filename in os.listdir(DATA_FOLDER):

        path = os.path.join(
            DATA_FOLDER,
            filename
        )

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

def create_chunks(
    documents,
    chunk_size=1000,
    overlap=200
):

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
            f"Embedded chunk "
            f"{i + 1}/{len(chunks)}"
        )

    return collection


# ==================================================
# 9. RAG SEARCH FUNCTION
# ==================================================

def search_documents(
    collection,
    question
):

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

    context = (
        "\n\n----------------\n\n"
        .join(context_parts)
    )

    return context


# ==================================================
# 10. SAFE CALCULATOR
# ==================================================

def calculate(expression):

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

        if isinstance(
            node,
            ast.Constant
        ):

            if isinstance(
                node.value,
                (int, float)
            ):

                return node.value

            raise ValueError(
                "Only numbers are allowed."
            )

        elif isinstance(
            node,
            ast.BinOp
        ):

            left = evaluate(
                node.left
            )

            right = evaluate(
                node.right
            )

            operation = allowed_operators.get(
                type(node.op)
            )

            if operation is None:

                raise ValueError(
                    "Operator not allowed."
                )

            return operation(
                left,
                right
            )

        elif isinstance(
            node,
            ast.UnaryOp
        ):

            value = evaluate(
                node.operand
            )

            operation = allowed_operators.get(
                type(node.op)
            )

            if operation is None:

                raise ValueError(
                    "Operator not allowed."
                )

            return operation(
                value
            )

        else:

            raise ValueError(
                "Invalid expression."
            )

    try:

        tree = ast.parse(
            expression,
            mode="eval"
        )

        result = evaluate(
            tree.body
        )

        return str(result)

    except Exception as e:

        return f"Calculation error: {e}"


# ==================================================
# 11. DATA ANALYSIS TOOL
# ==================================================

def analyze_data(
    filename,
    operation,
    column
):

    # ----------------------------------------------
    # Check file
    # ----------------------------------------------

    if filename not in dataframes:

        return (
            f"File '{filename}' is not available "
            f"for data analysis."
        )

    df = dataframes[filename]

    # ----------------------------------------------
    # Check column
    # ----------------------------------------------

    if column not in df.columns:

        return (
            f"Column '{column}' was not found "
            f"in {filename}."
        )

    # ----------------------------------------------
    # Convert column to numeric
    # ----------------------------------------------

    numeric_column = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    valid_values = numeric_column.dropna()

    if len(valid_values) == 0:

        return (
            f"Column '{column}' does not "
            f"contain numeric values."
        )

    # ----------------------------------------------
    # Operations
    # ----------------------------------------------

    if operation == "average":

        result = valid_values.mean()

        return (
            f"Average {column}: "
            f"{result:.2f}"
        )

    elif operation == "sum":

        result = valid_values.sum()

        return (
            f"Total {column}: "
            f"{result:.2f}"
        )

    elif operation == "minimum":

        result = valid_values.min()

        return (
            f"Minimum {column}: "
            f"{result:.2f}"
        )

    elif operation == "maximum":

        result = valid_values.max()

        return (
            f"Maximum {column}: "
            f"{result:.2f}"
        )

    elif operation == "count":

        result = valid_values.count()

        return (
            f"Count of {column}: "
            f"{result}"
        )

    else:

        return (
            f"Unsupported operation: "
            f"{operation}"
        )


# ==================================================
# 12. LIST DATASETS TOOL
# ==================================================

def list_datasets():

    if not dataframes:

        return "No Excel or CSV files loaded."

    result = []

    for filename, df in dataframes.items():

        columns = list(df.columns)

        result.append(

            f"File: {filename}\n"
            f"Columns: {columns}\n"
            f"Rows: {len(df)}"

        )

    return "\n\n".join(result)


# ==================================================
# 13. TOOL DEFINITIONS
# ==================================================

calculator_tool = {

    "type": "function",

    "function": {

        "name": "calculate",

        "description": (
            "Perform mathematical calculations. "
            "Use for expressions such as "
            "25% of 80000, 100 + 200, "
            "500 * 20, or (1000 + 500) / 2."
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

            "required": [
                "expression"
            ]

        }

    }

}


data_analysis_tool = {

    "type": "function",

    "function": {

        "name": "analyze_data",

        "description": (
            "Analyze numeric columns in Excel "
            "or CSV files. Use this tool for "
            "average, sum, minimum, maximum, "
            "and count calculations."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "filename": {

                    "type": "string",

                    "description": (
                        "Exact Excel or CSV "
                        "filename."
                    )

                },

                "operation": {

                    "type": "string",

                    "enum": [

                        "average",
                        "sum",
                        "minimum",
                        "maximum",
                        "count"

                    ],

                    "description": (
                        "Operation to perform."
                    )

                },

                "column": {

                    "type": "string",

                    "description": (
                        "Exact numeric column "
                        "name."
                    )

                }

            },

            "required": [

                "filename",
                "operation",
                "column"

            ]

        }

    }

}


list_datasets_tool = {

    "type": "function",

    "function": {

        "name": "list_datasets",

        "description": (
            "List available Excel and CSV "
            "datasets and their columns. "
            "Use this before analyzing a dataset "
            "when the filename or column name "
            "is unknown."
        ),

        "parameters": {

            "type": "object",

            "properties": {}

        }

    }

}


# ==================================================
# 14. ALL TOOLS
# ==================================================

tools = [

    calculator_tool,

    data_analysis_tool,

    list_datasets_tool

]


# ==================================================
# 15. AGENT
# ==================================================

def ask_question(
    collection,
    question
):

    # ------------------------------------------------
    # RAG context
    # ------------------------------------------------

    context = search_documents(
        collection,
        question
    )

    # ------------------------------------------------
    # System instructions
    # ------------------------------------------------

    system_prompt = """
You are a company AI assistant.

You have three tools:

1. Calculator
   - For mathematical calculations.

2. Data Analysis Tool
   - For calculations on Excel/CSV data.
   - Use this for average, sum, minimum,
     maximum, and count.

3. Dataset Listing Tool
   - Use this to discover available
     Excel/CSV files and columns.

You also have document context from RAG.

Rules:

- Use RAG context for document questions.
- Use Calculator for normal mathematical
  calculations.
- Use Data Analysis Tool for Excel/CSV
  calculations.
- Never guess numerical results.
- Never calculate Excel/CSV statistics
  yourself when the Data Analysis Tool
  can perform them.
- If you need to know available files or
  columns, call list_datasets first.
- Give a simple final answer.
- Mention the source file when appropriate.
"""

    messages = [

        {
            "role": "system",

            "content": system_prompt
        },

        {
            "role": "user",

            "content": f"""
RAG context:

---------------- CONTEXT ----------------

{context}

---------------- QUESTION ----------------

{question}
"""
        }

    ]

    # ------------------------------------------------
    # Agent loop
    # ------------------------------------------------

    while True:

        response = ollama.chat(

            model=LLM_MODEL,

            messages=messages,

            tools=tools

        )

        # --------------------------------------------
        # No tool call
        # --------------------------------------------

        if not response.message.tool_calls:

            return response.message.content

        # --------------------------------------------
        # Add assistant response
        # --------------------------------------------

        messages.append(
            response.message
        )

        # --------------------------------------------
        # Execute tools
        # --------------------------------------------

        for tool_call in response.message.tool_calls:

            tool_name = (
                tool_call.function.name
            )

            arguments = (
                tool_call.function.arguments
            )

            print(
                f"\n[Tool Called] "
                f"{tool_name}"
            )

            print(
                f"Arguments: {arguments}"
            )

            # ----------------------------------------
            # Calculator
            # ----------------------------------------

            if tool_name == "calculate":

                expression = arguments[
                    "expression"
                ]

                result = calculate(
                    expression
                )

            # ----------------------------------------
            # Data Analysis
            # ----------------------------------------

            elif tool_name == "analyze_data":

                filename = arguments[
                    "filename"
                ]

                operation = arguments[
                    "operation"
                ]

                column = arguments[
                    "column"
                ]

                result = analyze_data(

                    filename,

                    operation,

                    column

                )

            # ----------------------------------------
            # Dataset listing
            # ----------------------------------------

            elif tool_name == "list_datasets":

                result = list_datasets()

            # ----------------------------------------
            # Unknown tool
            # ----------------------------------------

            else:

                result = (
                    f"Unknown tool: "
                    f"{tool_name}"
                )

            print(
                f"[Tool Result] {result}"
            )

            # ----------------------------------------
            # Send tool result back to LLM
            # ----------------------------------------

            messages.append({

                "role": "tool",

                "content": result

            })


# ==================================================
# 16. MAIN PROGRAM
# ==================================================

print("\n====================================")
print("RAG + TOOLS AI ASSISTANT")
print("====================================")


# --------------------------------------------------
# Load files
# --------------------------------------------------

documents = load_all_files()

print(
    f"\nTotal files loaded: "
    f"{len(documents)}"
)


# --------------------------------------------------
# Show datasets
# --------------------------------------------------

print(
    f"Structured datasets loaded: "
    f"{len(dataframes)}"
)

for filename, df in dataframes.items():

    print(
        f"  - {filename}: "
        f"{len(df)} rows"
    )


# --------------------------------------------------
# Create chunks
# --------------------------------------------------

chunks = create_chunks(
    documents
)

print(
    f"\nTotal chunks created: "
    f"{len(chunks)}"
)


# --------------------------------------------------
# Create vector database
# --------------------------------------------------

collection = create_vector_database(
    chunks
)


# --------------------------------------------------
# Ready
# --------------------------------------------------

print("\n====================================")
print("AI ASSISTANT READY")
print("====================================")

print(
    "\nAvailable capabilities:"
)

print(
    "1. RAG"
)

print(
    "2. Calculator"
)

print(
    "3. Excel/CSV Data Analysis"
)


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

    try:

        answer = ask_question(
            collection,
            question
        )

        print(
            "\n===================================="
        )

        print("ANSWER")

        print(
            "===================================="
        )

        print(answer)

    except Exception as e:

        print(
            f"\nError: {e}"
        )