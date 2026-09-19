import os
from pathlib import Path

from dotenv import load_dotenv
from source_metadata import SOURCE_METADATA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# --------------------------------------------------
# 1. Load knowledge-base documents
# --------------------------------------------------

KB_PATH = Path("knowledge_base")
documents = []

for file_path in KB_PATH.glob("*.md"):
    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    loaded_docs = loader.load()

    file_sources = SOURCE_METADATA.get(file_path.name, [])

    for doc in loaded_docs:
        doc.metadata["source_file"] = file_path.name
        doc.metadata["sources"] = file_sources

    documents.extend(loaded_docs)

# --------------------------------------------------
# 2. Split documents into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)
print("\nFirst chunk metadata:")
print(chunks[0].metadata)
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
# 8. Helper function to format documents
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

def format_sources(docs):
    sources = {}

    for doc in docs:
        for source in doc.metadata.get("sources", []):
            sources[source["url"]] = source["title"]

    return [
        {
            "title": title,
            "url": url
        }
        for url, title in sources.items()
    ]

# --------------------------------------------------
# 9. Build LangChain RAG chain
# --------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# --------------------------------------------------
# 10. Ask a question
# --------------------------------------------------

question = "What can I do in Singapore if it rains?"
response = rag_chain.invoke(question)

sources = format_sources(
    retriever.invoke(question)
)

# --------------------------------------------------
# 11. Display answer
# --------------------------------------------------

if isinstance(response.content, list):
    answer = "\n".join(
        item["text"]
        for item in response.content
        if item.get("type") == "text"
    )
else:
    answer = response.content

print("\nANSWER")
print("=" * 80)
print(answer)

print("\nSOURCES")
print("=" * 80)

for source in sources:
    print(f"- {source['title']}")
    print(f"  {source['url']}")