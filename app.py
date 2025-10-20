from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import json
import re

# ============================================================
#  Gemini Configuration
# ============================================================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Use the latest, stable model
MODEL_NAME = "models/gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)
print(f"✅ Using Gemini model: {MODEL_NAME}")

# ============================================================
#  FastAPI App Setup
# ============================================================
app = FastAPI(title="Job Ad Cleaner Agent")

# ✅ Enable CORS (adjust origins for your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["https://your-frontend.onrender.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  Utility: Safe JSON Parser
# ============================================================
def safe_json_parse(text):
    """Attempts to safely parse model output into valid JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Remove markdown code fences and retry
        cleaned_text = re.sub(r"```(?:json)?|```", "", text).strip()
        try:
            return json.loads(cleaned_text)
        except Exception as e:
            print("⚠️ JSON repair failed:", e)
            return None

# ============================================================
#  Request Schema
# ============================================================
class AnalyzerInput(BaseModel):
    job_text: str
    raw_output: dict

# ============================================================
#  Core Endpoint: /clean
# ============================================================
@app.post("/clean")
def clean_output(payload: AnalyzerInput):
    prompt = f"""
You are a post-processing agent for a job fraud detection system.

### TASK
Clean and clarify the analyzer's raw JSON output.

### RULES
1. Maintain the same JSON structure.
2. Do NOT add any new words or phrases.
3. You may merge incomplete or truncated phrases using nearby text from the job ad.
4. Remove meaningless phrases (like "subject") if they don't exist in the text.
5. Keep the same phrase order as they appear in the job ad.
6. For each phrase, add a "reason" key explaining in one short sentence why it may indicate a fake job.
7. Return **only valid JSON** — no markdown, no extra text.

### JOB AD:
{payload.job_text}

### RAW OUTPUT:
{json.dumps(payload.raw_output, indent=2)}
"""

    try:
        # Ask Gemini to return strict JSON
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        cleaned = safe_json_parse(response.text)
        if not cleaned:
            raise HTTPException(status_code=500, detail="Model returned invalid JSON.")
        return cleaned

    except Exception as e:
        print("❌ Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
#  Diagnostics Endpoint: /models
# ============================================================
@app.get("/models")
def list_available_models():
    models = []
    for m in genai.list_models():
        models.append({
            "name": m.name,
            "methods": m.supported_generation_methods
        })
    return {"available_models": models}
