import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or "your_actual_key" in GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set or is using placeholder value.")

app = FastAPI(title="ScribeMind Backend")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SuggestionRequest(BaseModel):
    text: str
    tone: str

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/suggestions")
async def get_suggestions(request: SuggestionRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="API Key not configured on server.")

    prompt = f"""
    You are a real-time communication assistant. 
    Analyze the following transcript of a person speaking.
    Transcript: "{request.text}"
    Tone: {request.tone}

    Provide a structured JSON response with:
    1. "intent": A very short summary of what the user is trying to communicate (max 5 words).
    2. "suggestions": An array of 3 highly effective, concise response suggestions based on the tone.

    STRICT JSON ONLY. NO PROLOGUE OR EPILOGUE.
    Format:
    {{
        "intent": "string",
        "suggestions": ["str1", "str2", "str3"]
    }}
    """

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": os.getenv("GROQ_MODEL", "llama3-70b-8192"),
                    "messages": [
                        {"role": "system", "content": "You are a communication expert that outputs strict JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"}
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                print(f"Groq API Error: {response.text}")
                raise HTTPException(status_code=response.status_code, detail="Error from Groq API")

            data = response.json()
            content_str = data["choices"][0]["message"]["content"]
            
            # Additional safety: strip markdown if present
            if "```" in content_str:
                content_str = content_str.split("```json")[-1].split("```")[0].strip()
            
            return json.loads(content_str)

        except Exception as e:
            print(f"Server Error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
