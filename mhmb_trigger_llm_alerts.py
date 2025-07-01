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

def generate_combined_alert(anomalies, qa_chain):
    description_list = [
        "abnormal ECG" if k == "ecg_anomaly" else f"{k.replace('_', ' ')} = {v}"
        for k, v in anomalies.items()
    ]
    combined_description = "; ".join(description_list)
    query = (
        f"The patient shows the following anomalies: {combined_description}. "
        f"What might this indicate and what actions should be taken?"
    )
    response = qa_chain(query)
    sources = ", ".join(doc.metadata.get("source", "Unknown") for doc in response['source_documents'])

    return f"""/n Combined Alert 
Anomalies detected: {combined_description}

 LLM Medical Explanation:
{response['result']}

"""

def trigger_alert(message):
    print(message)

if __name__ == "__main__":
    patient_data = {
        "skin_temperature": 38.2,
        "heart_rate": 105,
        "blood_pressure_systolic": 130,
        "blood_pressure_diastolic": 90,
        "SpO2": 91,
        "mobility": 500,
        "ecg_anomaly": True
    }

    anomalies = check_anomalies(patient_data)
    
    if anomalies:
        alert_message = generate_combined_alert(anomalies, qa_chain)
        trigger_alert(alert_message)
