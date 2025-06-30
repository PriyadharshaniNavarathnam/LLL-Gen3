import openai
import json

# ✅ Initialize OpenAI client with your API key
client = openai.OpenAI(api_key="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")  # Replace with your real API key

# --- Config ---
input_file = "llm_prompts.txt"
output_file = "llm_responses.json"
model_name = "gpt-3.5-turbo"  # ✅ Use GPT-4 only if you have access

# --- Load prompts from file ---
with open(input_file, "r") as f:
    raw = f.read()

# --- Split prompts (each patient is separated by double newline) ---
prompts = [p.strip() for p in raw.strip().split("\n\n") if p.strip()]
responses = {}

# --- Loop through each prompt and query OpenAI ---
for prompt in prompts:
    patient_id_line = prompt.split("\n")[0]
    patient_id = patient_id_line.split(":")[1].strip()

    print(f"⏳ Processing Patient {patient_id}...")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a medical assistant providing clinical insights."},
                {"role": "user", "content": prompt}
            ]
        )

        interpretation = response.choices[0].message.content.strip()

        responses[patient_id] = {
            "prompt": prompt,
            "interpretation": interpretation
        }

        print(f"✅ Done for Patient {patient_id}")

    except Exception as e:
        print(f"❌ Error for Patient {patient_id}: {e}")
        responses[patient_id] = {"error": str(e)}

# --- Save all responses to a JSON file ---
with open(output_file, "w") as f:
    json.dump(responses, f, indent=2)

print(f"\n🎉 All interpretations saved to {output_file}")
