from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Location of our knowledge base
KB_PATH = Path("knowledge_base")

# 1. Load documents
documents = []

for file_path in KB_PATH.glob("*.md"):
    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )
    loaded_docs = loader.load()
    documents.extend(loaded_docs)
print(f"Loaded documents: {len(documents)}")

for doc in documents:
    print(f" - {doc.metadata['source']}")

# 2. Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

# 3. Split documents into chunks
chunks = text_splitter.split_documents(documents)
print(f"\nTotal chunks: {len(chunks)}")

# 4. Display chunks
for i, chunk in enumerate(chunks):
    print("\n" + "=" * 80)
    print(f"CHUNK {i + 1}")
    print(f"Source: {chunk.metadata['source']}")
    print("-" * 80)
    print(chunk.page_content)