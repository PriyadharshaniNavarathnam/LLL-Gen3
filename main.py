import os
import requests
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
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
assert GROQ_API_KEY, "Missing GROQ_API_KEY"
assert OPENWEATHER_API_KEY, "Missing OPENWEATHER_API_KEY"

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Normal ranges
NORMAL_RANGES = {
    "skin_temperature": (36.1, 37.2),
    "heart_rate": (60, 100),
    "blood_pressure_systolic": (90, 120),
    "blood_pressure_diastolic": (60, 80),
    "SpO2": (95, 100),
    "mobility": (1000, 20000),
}

# Hardcoded baseline (example)
BASELINE = {
    "skin_temperature": 36.5,
    "heart_rate": 75,
    "blood_pressure_systolic": 110,
    "blood_pressure_diastolic": 78,
    "SpO2": 98,
    "mobility": 8000,
}

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

class DailySuggestionRequest(BaseModel):
    username: str
    city: str
    skin_temperature: float
    heart_rate: int
    blood_pressure_systolic: int
    blood_pressure_diastolic: int
    SpO2: int
    mobility: int
    ecg_anomaly: bool

# Helpers
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

def get_weather_by_city(city: str) -> str:
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"Today in {city}, it is {temp}°C with {weather}."
    except:
        return "Weather data unavailable."

def simplify_explanation(text):
    lines = text.split('\n')
    filtered = [l for l in lines if not l.lower().startswith("i don't know") and l.strip() != ""]
    cleaned = []
    for line in filtered:
        if len(cleaned) >= 4:
            break
        line = line.replace("SpO2", "oxygen level").replace("tachycardic", "fast heart rate")
        cleaned.append(line.strip())
    cleaned.append("Please consult your doctor for proper care.")
    return " ".join(cleaned)

def get_preventive_advice(username: str, current: dict) -> str:
    advice_input = []
    for key in BASELINE:
        baseline = BASELINE[key]
        now = current.get(key)
        if now is not None:
            diff = abs(now - baseline)
            if diff >= 5 or (key == "mobility" and now < baseline * 0.5):
                advice_input.append(f"{key.replace('_', ' ').capitalize()}: {now} vs baseline {baseline}")

    if not advice_input:
        return ""

    joined = "; ".join(advice_input)
    prompt = f"Hello {username}, I'm concerned about the deviations in your vitals: {joined}. Suggest simple lifestyle changes to normalize them."
    result = qa_chain({"query": prompt})
    return simplify_explanation(result["result"])

# ALERTS
@app.post("/trigger-alert")
async def trigger_alert(vitals: VitalsData):
    data = vitals.dict()
    if all((v == 0 or v is False) for k, v in data.items() if k != "ecg_anomaly"):
        return {
            "alert_summary": "Device fault: No vitals detected.",
            "medical_explanation": "Please check the health monitoring device. All vitals are zero."
        }

    anomalies = check_anomalies(data)
    if not anomalies:
        return {"message": "All vitals are within normal range."}

    summary = "; ".join(["abnormal ECG" if k == "ecg_anomaly" else f"{k.replace('_',' ')} = {v}" for k, v in anomalies.items()])
    query = f"The patient shows the following anomalies: {summary}. What does this suggest and what actions should be taken?"
    result = qa_chain({"query": query})
    explanation = simplify_explanation(result["result"])

    preventive = get_preventive_advice("Patient", data)

    return {
        "alert_summary": summary,
        "medical_explanation": explanation,
        "preventive_advice": preventive or "No preventive advice required."
    }

@app.post("/ask")
async def ask(req: ChatRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    response = qa_chain({"query": q})
    return {"answer": simplify_explanation(response["result"]) }

@app.post("/daily-suggestion")
async def daily_suggestion(req: DailySuggestionRequest):
    vitals = {
        "skin_temperature": req.skin_temperature,
        "heart_rate": req.heart_rate,
        "blood_pressure_systolic": req.blood_pressure_systolic,
        "blood_pressure_diastolic": req.blood_pressure_diastolic,
        "SpO2": req.SpO2,
        "mobility": req.mobility,
        "ecg_anomaly": req.ecg_anomaly
    }

    weather = get_weather_by_city(req.city)
    prompt = (
        f"Hey! Here are the vitals for a patient named {req.username}: {vitals}. {weather} "
        f"Based on this information, suggest a warm, conversational daily health tip as if you're a friendly AI companion. "
        f"Keep the tone helpful and supportive, like: 'Hey there! On this {weather.lower()}, remember to...'"
    )
    response = qa_chain({"query": prompt})
    return {"username": req.username, "daily_tip": simplify_explanation(response["result"]) }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
