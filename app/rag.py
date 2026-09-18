import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from source_metadata import SOURCE_METADATA

# ------------------------------------------------------------
# Environment Configuration
# ------------------------------------------------------------
# Load environment variables from .env.
# The Gemini API key is kept outside the source code.
load_dotenv()


# ------------------------------------------------------------
# 1. Load Knowledge Base Documents
# ------------------------------------------------------------
# All Markdown files inside knowledge_base/ are treated as the
# application's stable destination knowledge.
#
# The KB currently contains information about:
# - attractions and neighbourhoods
# - transportation
# - culture, food and practical tips
# - sample itineraries
knowledge_base_path = Path("knowledge_base")

documents = []

for file_path in knowledge_base_path.glob("*.md"):
    # TextLoader converts each Markdown file into LangChain
    # Document objects.
    loader = TextLoader(
        file_path,
        encoding="utf-8",
    )

    loaded_docs = loader.load()

    # Retrieve the source information associated with this
    # particular KB file.
    #
    # Example:
    # attractions.md -> Wikivoyage + Visit Singapore
    file_sources = SOURCE_METADATA.get(
        file_path.name,
        [],
    )

    for doc in loaded_docs:

        # Store the KB filename as metadata.
        # This allows us to identify which internal document
        # produced a retrieved chunk.
        doc.metadata["source_file"] = file_path.name

        # Attach the public source information to the document.
        # This metadata will later be included in the final
        # LLM context so that source references can be displayed.
        doc.metadata["sources"] = file_sources

    documents.extend(loaded_docs)

# ------------------------------------------------------------
# 2. Split Documents into Meaningful Chunks
# ------------------------------------------------------------
# Large documents should not be sent to the LLM as one block.
#
# Instead, the documents are divided into smaller overlapping
# chunks so that semantic retrieval can find the portions that
# are most relevant to a user's question.
#
# chunk_size:
#     Maximum target size of each chunk.
#
# chunk_overlap:
#     Keeps some content from the previous chunk so that
#     information spanning two chunks is less likely to be lost.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(documents)

# ------------------------------------------------------------
# 3. Create Embeddings
# ------------------------------------------------------------
# Convert each text chunk into a numerical vector representation.
#
# Embeddings allow FAISS to perform semantic similarity search.
# This means retrieval is based on the meaning of the question,
# rather than requiring an exact keyword match.
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# ------------------------------------------------------------
# 4. Create FAISS Vector Store
# ------------------------------------------------------------
# Store the document embeddings in a FAISS vector index.
#
# When a user asks a question later, the question is also
# represented as an embedding and FAISS finds the most
# semantically similar document chunks.
vector_store = FAISS.from_documents(
    chunks,
    embeddings,
)

# ------------------------------------------------------------
# 5. Create the Retriever
# ------------------------------------------------------------
# Convert the vector store into a LangChain retriever.
#
# k=5 means the retriever returns the five most relevant
# chunks for each query.
#
# We tested transportation-specific and combined itinerary
# queries and confirmed that semantic retrieval returns
# relevant KB content.
retriever = vector_store.as_retriever(search_kwargs={"k": 5})


# ------------------------------------------------------------
# 6. Format Retrieved Documents with Source Information
# ------------------------------------------------------------
def format_docs_with_sources(docs):
    """
    Convert retrieved LangChain documents into text that can be
    included in the final LLM prompt.

    Each retrieved chunk contains:
        - the internal KB filename
        - public source title(s)
        - public source URL(s)
        - the actual retrieved content

    Including the source metadata in the LLM context allows the
    final response to provide source references for KB-derived
    information.
    """

    formatted_docs = []

    for doc in docs:

        # Retrieve metadata added when the KB was loaded.
        source_file = doc.metadata.get(
            "source_file",
            "Unknown",
        )

        sources = doc.metadata.get(
            "sources",
            [],
        )
        source_lines = []

        for source in sources:
            source_lines.append(f"- {source['title']} ({source['url']})")

        # Combine source information and the retrieved chunk
        # into a clearly separated block.
        formatted_docs.append(f"""
        [Source file: {source_file}]
        [Sources:]
        {chr(10).join(source_lines)}

        {doc.page_content}
        """)

    return "\n\n".join(formatted_docs)
