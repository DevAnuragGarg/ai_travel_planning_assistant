from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os


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
    documents.extend(loader.load())

print(f"Loaded documents: {len(documents)}")


# --------------------------------------------------
# 2. Split documents into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# --------------------------------------------------
# 3. Create embedding model
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
print("FAISS vector store created.")

# --------------------------------------------------
# 5. Search the vector store
# --------------------------------------------------

query = "What can I do in Singapore if it rains?"

results = vector_store.similarity_search(
    query,
    k=3
)

# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\nTop matching chunks:\n")
for i, document in enumerate(results, start=1):
    print("=" * 80)
    print(f"RESULT {i}")
    print(f"Source: {document.metadata['source']}")
    print("-" * 80)
    print(document.page_content)