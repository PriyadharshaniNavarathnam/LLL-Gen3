import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from preprocess_books import get_embeddings

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "Missing GROQ_API_KEY in .env"

# Load Pinecone retriever
retriever = PineconeVectorStore.from_existing_index(
    index_name="mhmb",
    embedding=get_embeddings()
).as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Use Groq with LLaMA 3
llm = ChatGroq(
    model_name="llama3-70b-8192"  # You can change to "llama3-70b-8192" if needed
)

# Set up Retrieval-based QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Ask a question
query = "What does low blood oxygen level (SpO2) mean and what should a patient do?"
response = qa_chain(query)

# Output
print("\n LLM Response:\n", response['result'])


