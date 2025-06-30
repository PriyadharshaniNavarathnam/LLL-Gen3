import os
import pandas as pd
from glob import glob

# --- Paths ---
input_folder = "processed_csv"
output_file = "llm_prompts.txt"

# --- Metrics to extract (no spaces now, we'll strip columns later) ---
metrics = ["SpO2", "HR", "RESP", "PULSE"]

# --- Choose method: 'average' or 'latest' ---
method = "average"

# --- Helper function ---
def extract_summary(df, method="average"):
    summary = {}
    for col in metrics:
        if col in df.columns:
            if method == "average":
                summary[col] = round(df[col].mean(), 2)
            elif method == "latest":
                summary[col] = df[col].dropna().iloc[-1]
    return summary

# --- Main script ---
with open(output_file, "w") as out_file:
    for file_path in glob(os.path.join(input_folder, "*_Processed.csv")):
        df = pd.read_csv(file_path)

        # --- Clean column names (strip leading/trailing spaces) ---
        df.columns = [col.strip() for col in df.columns]

        # --- Extract subject ID from file name ---
        subject_id = os.path.basename(file_path).split("_")[1]

        # --- Get Age and Gender ---
        age = df["Age"].dropna().iloc[0] if "Age" in df.columns else "Unknown"
        gender = df["Gender"].dropna().iloc[0] if "Gender" in df.columns else "Unknown"

        # --- Get summary of metrics ---
        summary = extract_summary(df, method)

        # --- Format the prompt ---
        prompt = (
            f"Patient ID: {subject_id}\n"
            f"Age: {age}\n"
            f"Gender: {gender}\n"
            f"Health metrics:\n"
        )
        for key, val in summary.items():
            prompt += f"- {key}: {val}\n"
        prompt += "Provide a medical interpretation.\n\n"

        out_file.write(prompt)
        print(f"✅ Prompt generated for Patient {subject_id}")
