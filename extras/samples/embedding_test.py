import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# chunked string
text = "Cloud Forest is an indoor attraction in Singapore."

# create vector embedding of chunked string
vector = embeddings.embed_query(text)

print("Vector dimensions:", len(vector))
print("First 10 values:", vector[:10])

texts = [
    "Cloud Forest is an indoor attraction in Singapore.",
    "What can I do in Singapore when it rains?",
    "The MRT is a convenient way to travel around Singapore."
]

vectors = embeddings.embed_documents(texts)

for text, vector in zip(texts, vectors):
    print("\nText:", text)
    print("Vector dimensions:", len(vector))
    print("First 5 values:", vector[:5])