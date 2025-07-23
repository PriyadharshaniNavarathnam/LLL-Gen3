import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from preprocess_books import get_embeddings

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "Missing GROQ_API_KEY in .env"

# Create FastAPI app
app = FastAPI()

# ✅ CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to ["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Normal ranges for vitals
NORMAL_RANGES = {
    "skin_temperature": (36.1, 37.2),
    "heart_rate": (60, 100),
    "blood_pressure_systolic": (90, 120),
    "blood_pressure_diastolic": (60, 80),
    "SpO2": (95, 100),
    "mobility": (1000, 20000),
}

# Load vector retriever and LLM
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

# Request models
class VitalsData(BaseModel):
    skin_temperature: float
    heart_rate: int
    blood_pressure_systolic: int
    blood_pressure_diastolic: int
    SpO2: int
    mobility: int
    ecg_anomaly: bool

class ChatRequest(BaseModel):
    question: str

# Anomaly checker
def check_anomalies(data):
    anomalies = {}
    for key, value in data.items():
        if key == "ecg_anomaly" and value is True:
            anomalies[key] = "Abnormal ECG"
        elif key in NORMAL_RANGES:
            low, high = NORMAL_RANGES[key]
            if value < low or value > high:
                anomalies[key] = value
    return anomalies

# Simplify LLM response for patient
def simplify_explanation(text: str) -> str:
    lines = text.split('\n')
    simplified_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("*") or line[0].isdigit() or len(simplified_lines) < 4:
            line = line.replace('"', '').replace('SpO2', 'oxygen level').replace('tachycardic', 'fast heart rate')
            simplified_lines.append(line)
        if len(simplified_lines) >= 4:
            break
    simplified_lines.append("Please consult your doctor for proper care.")
    return " ".join(simplified_lines)

# 🚨 Anomaly alert endpoint
@app.post("/trigger-alert")
async def trigger_alert(vitals: VitalsData):
    patient_data = vitals.dict()

    all_zero = all(
        (value == 0 or value is False)
        for key, value in patient_data.items()
        if key != "ecg_anomaly"
    )
    if all_zero:
        return {
            "alert_summary": "Invalid or missing sensor data detected.",
            "medical_explanation": (
                "All vital readings are zero, which likely indicates sensor error or missing data. "
                "Please check the health monitoring device for proper operation."
            )
        }

    anomalies = check_anomalies(patient_data)
    if not anomalies:
        return {"message": "All vitals are within normal range."}

    description = [
        "abnormal ECG" if k == "ecg_anomaly" else f"{k.replace('_', ' ')} = {v}"
        for k, v in anomalies.items()
    ]
    alert_summary = "; ".join(description)

    query = (
        f"The patient shows the following anomalies: {alert_summary}. "
        f"What does this suggest and what actions should be taken?"
    )
    response = qa_chain({"query": query})
    explanation = response["result"]

    simplified = simplify_explanation(explanation)

    return {
        "alert_summary": alert_summary,
        "medical_explanation": simplified
    }

# 💬 Chatbot endpoint
@app.post("/ask")
async def ask_question(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    response = qa_chain({"query": question})
    return {"answer": response["result"]}

# Run locally (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
