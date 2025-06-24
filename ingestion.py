import os
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

google_api_key = os.environ["GOOGLE_API_KEY"]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/langchain-docs", encoding="utf-8")
    raw_documents = loader.load()
    print(f"📄 Loaded {len(raw_documents)} raw documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(raw_documents)
    print(f"✂️ Split into {len(split_docs)} chunks")

    # Clean URLs
    for doc in split_docs:
        new_url = doc.metadata["source"].replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    # Upload in small batches to avoid 4MB limit
    BATCH_SIZE = 50
    index_name = "langchain-doc-index"
    print("⬆️ Uploading to Pinecone...")

    vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)
    
    for i in range(0, len(split_docs), BATCH_SIZE):
        batch = split_docs[i:i + BATCH_SIZE]
        try:
            vectorstore.add_documents(batch)
        except Exception as e:
            print(f"⚠️ Failed to upload batch {i // BATCH_SIZE + 1}: {e}")
            continue

    print("✅ Upload complete!")

if __name__ == "__main__":
    ingest_docs()
