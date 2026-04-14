import os
import re
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY or "your_actual_key" in GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set or is using a placeholder value.")
else:
    logger.info(f"Groq API key loaded. Using model: {GROQ_MODEL}")

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
    system_prompt: str


def _extract_json(text: str) -> dict | None:
    """
    Robustly extract a JSON object from a model response string.
    Handles:
      - Pure JSON responses
      - Markdown-fenced JSON (```json ... ``` or ``` ... ```)
      - JSON embedded inside extra prose
    Returns a parsed dict, or None if no valid JSON is found.
    """
    # 1. Try direct parse first (happy path)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences (```json ... ``` or ``` ... ```)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Find the first {...} block anywhere in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None

@app.get("/")
def serve_ui():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/suggestions")
async def get_suggestions(request: SuggestionRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="API Key not configured on server.")

    # The system prompt is passed from the frontend. We append formatting instructions.
    full_system_prompt = (
        f"{request.system_prompt}\n\n"
        f"The user's current tone preference is: {request.tone}.\n"
        "STRICT INSTRUCTION: Respond ONLY with a valid JSON object. No explanation, no markdown blocks, no text before or after."
        "The JSON MUST follow this schema:\n"
        "{\n"
        '  "intent": "string (max 5 words)",\n'
        '  "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]\n'
        "}"
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": f"Conversation Context/Transcript:\n{request.text}"}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }

    logger.info(f"Sending request to Groq API | model={GROQ_MODEL} | context_length={len(request.text)}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30.0
            )

            if response.status_code != 200:
                logger.error(f"Groq API returned {response.status_code}: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Groq API error (status {response.status_code})"
                )

            data = response.json()
            content_str = data["choices"][0]["message"]["content"]
            logger.info(f"Raw Groq response received.")

            # Extract JSON robustly — handles markdown fences and leading/trailing text
            parsed = _extract_json(content_str)
            if parsed is None:
                logger.error(f"Failed to parse JSON from Groq response: {content_str!r}")
                raise HTTPException(status_code=500, detail="Model returned non-JSON response.")

            return parsed

        except HTTPException:
            raise
        except httpx.TimeoutException:
            logger.error("Request to Groq API timed out.")
            raise HTTPException(status_code=504, detail="Groq API request timed out.")
        except Exception as e:
            logger.exception(f"Unexpected server error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
