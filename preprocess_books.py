#reuired libraries
#pip install langchain pypdf pinecone-client sentence-transformers langchain-pinecone




import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
assert PINECONE_API_KEY, "PINECONE_API_KEY missing in .env"

# 1. Load PDFs
def load_documents(folder):
    loader = DirectoryLoader(folder, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

# 2. Split text into chunks
def split_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# 3. Load embedding model
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Upload to Pinecone
def upload_to_pinecone(text_chunks, index_name="mhmb"):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    embeddings = get_embeddings()
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=index_name
    )

if __name__ == "__main__":
    print("📚 Loading medical PDFs...")
    docs = load_documents("Data/")
    print(f"📄 Loaded {len(docs)} documents.")

    print("✂️ Splitting into chunks...")
    chunks = split_chunks(docs)
    print(f"✅ Created {len(chunks)} text chunks.")

    print("📡 Uploading to Pinecone...")
    upload_to_pinecone(chunks)
    print("🎉 Done: Your medical books are now searchable.")
