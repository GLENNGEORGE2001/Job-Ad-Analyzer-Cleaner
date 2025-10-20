from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import json
import os

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-pro")
app = FastAPI(title="Job Ad Cleaner Agent")

# Request schema
class AnalyzerInput(BaseModel):
    job_text: str
    raw_output: dict

@app.post("/clean")
def clean_output(payload: AnalyzerInput):
    prompt = f"""
You are a post-processing agent for a job fraud detector.

### INSTRUCTIONS:
- Maintain identical JSON structure.
- Do NOT add new words.
- You may merge incomplete phrases using nearby context.
- Remove meaningless phrases like "subject".
- Keep the same phrase order as in the job text.
- For each phrase, add a one-sentence "reason" explaining why it could indicate a fake job.
- Return valid JSON only.

### Job Ad:
{payload.job_text}

### Raw Output:
{json.dumps(payload.raw_output, indent=2)}

Now return the improved JSON.
"""

    try:
        response = model.generate_content(prompt)
        cleaned = json.loads(response.text)
        return cleaned
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
