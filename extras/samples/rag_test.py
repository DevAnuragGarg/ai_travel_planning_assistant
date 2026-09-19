import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# --------------------------------------------------
# 1. Load documents
# --------------------------------------------------

KB_PATH = Path("knowledge_base")
documents = []

for file_path in KB_PATH.glob("*.md"):
    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )
    documents.extend(loader.load())

# --------------------------------------------------
# 2. Split documents
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)
print(f"Loaded documents: {len(documents)}")
print(f"Total chunks: {len(chunks)}")

# --------------------------------------------------
# 3. Create embeddings
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --------------------------------------------------
# 4. Create FAISS vector store
# --------------------------------------------------

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

# --------------------------------------------------
# 5. Create retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# --------------------------------------------------
# 6. Create Gemini LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# --------------------------------------------------
# 7. Create prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a Singapore travel assistant.

Answer the user's question using ONLY the
knowledge-base context provided below.

If the context does not contain enough information
to answer the question, clearly say that the
knowledge base does not contain enough information.

Do not invent facts.

Knowledge-base context:
-----------------------
{context}
-----------------------

User question:
{question}

Provide a concise and useful answer.
"""
)

# --------------------------------------------------
# 8. Ask the user a question
# --------------------------------------------------

question = "What can I do in Singapore if it rains?"

# --------------------------------------------------
# 9. Retrieve relevant documents
# --------------------------------------------------

retrieved_docs = retriever.invoke(question)

# --------------------------------------------------
# 10. Build context
# --------------------------------------------------

context = "\n\n".join(
    document.page_content
    for document in retrieved_docs
)

# --------------------------------------------------
# 11. Create prompt
# --------------------------------------------------

messages = prompt.format_messages(
    context=context,
    question=question
)

# --------------------------------------------------
# 12. Ask Gemini
# --------------------------------------------------

response = llm.invoke(messages)

# --------------------------------------------------
# 13. Display answer
# --------------------------------------------------

print("\nANSWER")
print("=" * 80)
if isinstance(response.content, list):
    answer = "\n".join(
        item["text"]
        for item in response.content
        if item.get("type") == "text"
    )
else:
    answer = response.content

print(answer)

# --------------------------------------------------
# 14. Display sources
# --------------------------------------------------

print("\nSOURCES")
print("=" * 80)

for document in retrieved_docs:
    print("-", document.metadata["source"])