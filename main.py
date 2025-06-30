from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from dotenv import load_dotenv
from preprocess_books import get_embeddings
from mhmb_trigger_llm_alerts import check_anomalies, generate_combined_alert
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
import os

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Input model for vitals
class VitalsInput(BaseModel):
    skin_temperature: float
    heart_rate: float
    blood_pressure_systolic: float
    blood_pressure_diastolic: float
    SpO2: float
    mobility: int
    ecg_anomaly: bool

# Initialize retriever and LLM once
retriever = PineconeVectorStore.from_existing_index(
    index_name="mhmb",
    embedding=get_embeddings()
).as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(model_name="llama3-70b-8192")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

@app.post("/analyze")
async def analyze_vitals(vitals: VitalsInput, response: Response):
    data = vitals.dict()
    anomalies = check_anomalies(data)

    if not anomalies:
        response.status_code = status.HTTP_204_NO_CONTENT
        return

    qa_chain = get_chain()
    alert_message = generate_combined_alert(anomalies, qa_chain)
    return {
        "status": "alert",
        "message": alert_message
    }
