import os
import json
import base64
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
import edge_tts

# Load environment variables securely
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# Define Models
class Correction(BaseModel):
    user_text: str
    corrected_text: str
    explanation: str

class TutorResponse(BaseModel):
    reply_cebuano: str
    reply_english: str
    mistakes_found: bool
    corrections: list[Correction]

class UserMessage(BaseModel):
    message: str

class TTSRequest(BaseModel):
    text: str

# FastAPI App
app = FastAPI(title="Manang Tess Cebuano Tutor")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize GenAI Client
client = genai.Client()

tutor_config = types.GenerateContentConfig(
    system_instruction=(
        "You are a warm, encouraging Cebuano language tutor named Manang Tess. "
        "The user is a beginner. Respond primarily in simple Metro Cebu Bisaya. "
        "Always evaluate the user's last message for grammatical, lexical, and focus-affix errors."
        "Pay special attention to Austronesian focus-affix errors (mo-, nag-, gi-, -on, i-). "
        "You must return your response strictly matching the required JSON schema. The explanation in class Correction should explain the error and the correction to an English speaker who is learning Cebuano."
    ),
    temperature=0.3,
    response_mime_type="application/json",
    response_schema=TutorResponse,
)

# Global chat session (single-user design)
chat_session = None

async def generate_audio_base64(text: str) -> str:
    """Generates audio using edge-tts and returns it as a Base64 data URL."""
    communicate = edge_tts.Communicate(text, "fil-PH-BlessicaNeural")
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    
    b64_audio = base64.b64encode(audio_data).decode("utf-8")
    return f"data:audio/mp3;base64,{b64_audio}"

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

@app.post("/api/start")
async def start_lesson():
    global chat_session
    chat_session = client.chats.create(
        model="gemini-3.5-flash-lite", # Falls back gracefully if using standard APIs
        config=tutor_config
    )
    
    response = chat_session.send_message("Hello, I am ready to start my Cebuano lesson.")
    tutor_data = json.loads(response.text)
    
    # Generate TTS audio
    audio_url = await generate_audio_base64(tutor_data['reply_cebuano'])
    tutor_data['audio_base64'] = audio_url
    
    return tutor_data

@app.post("/api/chat")
async def send_message(user_msg: UserMessage):
    global chat_session
    if not chat_session:
        # Failsafe if user refreshed the page and continued chatting
        chat_session = client.chats.create(
            model="gemini-3.5-flash-lite",
            config=tutor_config
        )
        
    response = chat_session.send_message(user_msg.message)
    tutor_data = json.loads(response.text)
    
    # Generate TTS audio
    audio_url = await generate_audio_base64(tutor_data['reply_cebuano'])
    tutor_data['audio_base64'] = audio_url
    
    return tutor_data

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    audio_url = await generate_audio_base64(req.text)
    return {"audio_base64": audio_url}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
